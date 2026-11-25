"""Model, class weights, weighted loss, metrics and per-epoch reporting.

Model: MobileNetV2 pretrained on ImageNet, with the base FROZEN (feature extraction).
Only the classifier head is trained:
    global average pooling -> dropout -> dense (hidden units, L2) -> dropout -> dense (classes, softmax, L2).
Loss: class-weighted categorical cross-entropy (weights are inverse class frequencies, normalised to mean 1).
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tensorflow.keras import layers, regularizers

from src.preprocessing import IMG_SIZE

BASE_MODEL_NAME = "mobilenetv2_base"
CLASSIFIER_NAME = "tomato_mobilenetv2"


# ---------- class weights and loss ----------

def balanced_class_weights(class_counts):
    """Inverse-frequency weights scaled so that their mean is 1. Equal class sizes give a weight of 1 everywhere."""
    class_counts = np.asarray(class_counts, dtype=np.float64)
    number_of_classes = len(class_counts)
    raw_weights = class_counts.sum() / (number_of_classes * np.maximum(class_counts, 1))
    return (raw_weights / raw_weights.mean()).tolist()


def weighted_categorical_crossentropy(class_weights):
    """Categorical cross-entropy where every photo's loss is multiplied by the weight of its true class."""
    weight_tensor = tf.constant(class_weights, dtype=tf.float32)

    def loss(true_one_hot, predicted_probabilities):
        predicted_probabilities = tf.clip_by_value(predicted_probabilities, 1e-7, 1.0 - 1e-7)
        cross_entropy = -tf.reduce_sum(true_one_hot * tf.math.log(predicted_probabilities), axis=-1)
        photo_weight = tf.reduce_sum(true_one_hot * weight_tensor, axis=-1)
        return photo_weight * cross_entropy

    loss.__name__ = "weighted_categorical_crossentropy"
    return loss


# ---------- model ----------

def build_model(number_of_classes, dropout_rate=0.5, l2_strength=1e-4, learning_rate=1e-3,
                class_weights=None, hidden_units=128):
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False,
                                                   weights="imagenet", name=BASE_MODEL_NAME)
    base_model.trainable = False
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = layers.Rescaling(1 / 127.5, offset=-1)(inputs)      # MobileNetV2 expects pixels in [-1, 1]
    x = base_model(x, training=False)                        # BatchNorm stays in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(hidden_units, activation="relu", kernel_initializer="he_normal",
                     kernel_regularizer=regularizers.l2(l2_strength))(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(number_of_classes, activation="softmax",
                           kernel_regularizer=regularizers.l2(l2_strength))(x)
    model = tf.keras.Model(inputs, outputs, name=CLASSIFIER_NAME)
    compile_model(model, learning_rate=learning_rate, class_weights=class_weights)
    return model


def compile_model(model, learning_rate=1e-3, class_weights=None):
    if class_weights is None:
        class_weights = [1.0] * model.output_shape[-1]
    model.get_layer(BASE_MODEL_NAME).trainable = False      # the pretrained base is never trained
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=weighted_categorical_crossentropy(class_weights),
                  metrics=["accuracy"])
    return model


def load_trained(model_path):
    """Load a saved model for inference or fine-tuning. The loss is rebuilt by compile_model."""
    return tf.keras.models.load_model(model_path, compile=False)


# ---------- metrics ----------

def compute_metrics(true_labels, predicted_probabilities, class_weights):
    """Overall metrics: accuracy, weighted loss, macro precision, macro recall, macro F1 and macro one-vs-rest ROC AUC.

    The loss is computed from the predicted probabilities, so it does not include the L2 penalty that
    model.evaluate() adds. Macro averages give every class the same weight, so a weak class cannot hide.
    """
    true_labels = np.asarray(true_labels)
    predicted_probabilities = np.asarray(predicted_probabilities, dtype=np.float64)
    number_of_classes = predicted_probabilities.shape[1]
    predicted_labels = predicted_probabilities.argmax(axis=1)
    clipped = np.clip(predicted_probabilities, 1e-7, 1 - 1e-7)
    cross_entropy = -np.log(clipped[np.arange(len(true_labels)), true_labels])
    loss = float((np.asarray(class_weights)[true_labels] * cross_entropy).mean())
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="macro", labels=list(range(number_of_classes)), zero_division=0)
    try:
        roc_auc = float(roc_auc_score(true_labels, predicted_probabilities, multi_class="ovr",
                                      average="macro", labels=list(range(number_of_classes))))
    except ValueError:      # a class missing from the true labels makes ROC AUC undefined; None keeps JSON valid
        roc_auc = None
    return {"accuracy": float((predicted_labels == true_labels).mean()), "loss": loss,
            "precision": float(precision), "recall": float(recall), "f1": float(f1_score), "roc_auc": roc_auc}


def compute_per_class_metrics(true_labels, predicted_probabilities, class_names):
    """One row per class: precision, recall, F1 score, number of test photos, and one-vs-rest ROC AUC."""
    true_labels = np.asarray(true_labels)
    predicted_probabilities = np.asarray(predicted_probabilities)
    predicted_labels = predicted_probabilities.argmax(axis=1)
    number_of_classes = len(class_names)
    precision, recall, f1_score, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=list(range(number_of_classes)), zero_division=0)
    rows = []
    for class_index, class_name in enumerate(class_names):
        is_this_class = (true_labels == class_index).astype(int)
        try:
            class_roc_auc = float(roc_auc_score(is_this_class, predicted_probabilities[:, class_index]))
        except ValueError:
            class_roc_auc = None
        rows.append({"class": class_name, "precision": float(precision[class_index]),
                     "recall": float(recall[class_index]), "f1_score": float(f1_score[class_index]),
                     "test_photos": int(support[class_index]), "roc_auc": class_roc_auc})
    return rows


def evaluate_dataset(model, dataset, true_labels, class_weights):
    """Predict an unshuffled dataset and compute compute_metrics. true_labels must follow the dataset order."""
    predicted_probabilities = model.predict(dataset, verbose=0)
    return compute_metrics(true_labels, predicted_probabilities, class_weights)


# ---------- per-epoch reporting ----------

def _text(value):
    return "    n/a" if value is None else f"{value:.3f}"


class EpochMetrics(tf.keras.callbacks.Callback):
    """After every epoch, measures precision, recall, F1 score and ROC AUC on the training photos (no augmentation,
    inference mode) and on the validation photos, and also records the Keras validation accuracy and loss.

    The results are added to the Keras logs under readable keys, so TensorBoard and the training history receive them:
        precision, recall, f1_score, roc_auc                 (training photos)
        val_precision, val_recall, val_f1_score, val_roc_auc (validation photos)
    Keras' own loss, accuracy, val_loss and val_accuracy are not changed, so EarlyStopping and
    ReduceLROnPlateau keep watching the same values.
    """

    def __init__(self, training_dataset, training_labels, validation_dataset, validation_labels, class_weights,
                 on_epoch_finished=None, print_each_epoch=True):
        super().__init__()
        self.training_dataset = training_dataset
        self.training_labels = training_labels
        self.validation_dataset = validation_dataset
        self.validation_labels = validation_labels
        self.class_weights = class_weights
        self.on_epoch_finished = on_epoch_finished
        self.print_each_epoch = print_each_epoch
        self.history_rows = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        training_metrics = evaluate_dataset(self.model, self.training_dataset, self.training_labels, self.class_weights)
        validation_metrics = evaluate_dataset(self.model, self.validation_dataset, self.validation_labels,
                                              self.class_weights)
        row = {"epoch": epoch + 1,
               "loss": logs.get("loss"), "accuracy": logs.get("accuracy"),
               "precision": training_metrics["precision"], "recall": training_metrics["recall"],
               "f1_score": training_metrics["f1"], "roc_auc": training_metrics["roc_auc"],
               "val_loss": logs.get("val_loss"), "val_accuracy": logs.get("val_accuracy"),
               "val_precision": validation_metrics["precision"], "val_recall": validation_metrics["recall"],
               "val_f1_score": validation_metrics["f1"], "val_roc_auc": validation_metrics["roc_auc"]}
        self.history_rows.append(row)
        for key in ["precision", "recall", "f1_score", "roc_auc",
                    "val_precision", "val_recall", "val_f1_score", "val_roc_auc"]:
            logs[key] = float("nan") if row[key] is None else row[key]     # Keras logs must be numbers
        if self.print_each_epoch:
            print(f"epoch {epoch + 1:>2} | "
                  f"training: loss {_text(row['loss'])}  accuracy {_text(row['accuracy'])}  "
                  f"precision {_text(row['precision'])}  recall {_text(row['recall'])}  "
                  f"f1 score {_text(row['f1_score'])}  roc auc {_text(row['roc_auc'])} | "
                  f"validation: loss {_text(row['val_loss'])}  accuracy {_text(row['val_accuracy'])}  "
                  f"precision {_text(row['val_precision'])}  recall {_text(row['val_recall'])}  "
                  f"f1 score {_text(row['val_f1_score'])}  roc auc {_text(row['val_roc_auc'])}")
        if self.on_epoch_finished:
            self.on_epoch_finished(row)
