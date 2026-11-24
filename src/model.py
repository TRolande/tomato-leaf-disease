"""Model definition, loading and evaluation (custom CNN, no pre-trained weights)."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers

from src.preprocessing import IMG_SIZE


def build_optimized(num_classes, dropout=0.5, l2=1e-4, lr=1e-3):
    """4 conv blocks, He init, L2 regularization, Dropout and Adam. Same design as the notebook."""
    reg = regularizers.l2(l2)

    def conv_block(filters):
        return [layers.Conv2D(filters, 3, padding="same", activation="relu",
                              kernel_initializer="he_normal", kernel_regularizer=reg),
                layers.MaxPooling2D()]

    model = tf.keras.Sequential(
        [layers.Input(shape=IMG_SIZE + (3,)), layers.Rescaling(1.0 / 255)]
        + conv_block(32) + conv_block(64) + conv_block(128) + conv_block(128)
        + [layers.Flatten(), layers.Dropout(dropout),
           layers.Dense(128, activation="relu", kernel_initializer="he_normal", kernel_regularizer=reg),
           layers.Dropout(dropout),
           layers.Dense(num_classes, activation="softmax")],
        name="optimized_cnn")
    compile_model(model, lr)
    return model


def compile_model(model, lr=1e-3):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def load_trained(path):
    """Load a saved .h5 model for inference or fine-tuning."""
    return tf.keras.models.load_model(path, compile=False)


def evaluate_model(model, ds):
    """Loss, accuracy, precision, recall and F1 (weighted over classes) on a dataset."""
    from sklearn.metrics import precision_recall_fscore_support
    y_true, y_pred, losses = [], [], []
    cce = tf.keras.losses.CategoricalCrossentropy(reduction="none")
    for x, y in ds:
        probs = model(x, training=False).numpy()
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(probs, axis=1))
        losses.extend(cce(y, probs).numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": float((y_true == y_pred).mean()), "loss": float(np.mean(losses)),
            "precision": float(p), "recall": float(r), "f1": float(f1)}
