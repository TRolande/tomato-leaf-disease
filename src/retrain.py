"""Retraining pipeline used by the web app: uploads -> training folder -> preprocessing -> fine-tune -> evaluate -> new version.

It uses exactly the same functions as the training notebook (src/preprocessing.py and src/model.py):
the same decoding and resizing, the same augmentation, the same hash-based train/validation split, the same
class-weighted loss and the same metrics. Only the head is fine-tuned, with a small learning rate.
"""
import os
import shutil

import tensorflow as tf

from src.model import (EpochMetrics, balanced_class_weights, compile_model, evaluate_dataset, load_trained)
from src.preprocessing import count_per_class, labelled_items, make_dataset, split_train_validation

RETRAIN_LEARNING_RATE = 1e-4
RETRAIN_PATIENCE = 3


def retrain(database, model_service, training_folder, test_folder, models_folder, epochs=8,
            learning_rate=RETRAIN_LEARNING_RATE, on_progress=None):
    """Fine-tune the current model's head on every pending upload stored in the database."""
    report_progress = on_progress or (lambda **details: None)

    pending_uploads = database.pending_uploads()
    if not pending_uploads:
        raise ValueError("No new uploaded photos to retrain on. Upload some photos first.")

    class_names = list(model_service.class_names)
    number_of_classes = len(class_names)

    report_progress(message=f"Adding {len(pending_uploads)} uploaded photos to the training data")
    for upload in pending_uploads:
        class_folder = os.path.join(training_folder, upload["label"])
        os.makedirs(class_folder, exist_ok=True)
        copied_path = os.path.join(class_folder, f"new_{upload['id']}_{os.path.basename(upload['stored_path'])}")
        if not os.path.exists(copied_path):
            shutil.copy(upload["stored_path"], copied_path)

    all_training_items = labelled_items(training_folder, class_names)
    training_items, validation_items = split_train_validation(all_training_items, training_folder)
    test_items = labelled_items(test_folder, class_names)
    class_weights = balanced_class_weights(count_per_class(all_training_items, number_of_classes))

    report_progress(message="Preparing the photos")
    training_dataset = make_dataset(training_items, number_of_classes, shuffle=True, augment=True)
    validation_dataset = make_dataset(validation_items, number_of_classes)
    training_eval_dataset = make_dataset(training_items, number_of_classes)
    test_dataset = make_dataset(test_items, number_of_classes)
    training_labels = [class_index for _, class_index in training_items]
    validation_labels = [class_index for _, class_index in validation_items]
    test_labels = [class_index for _, class_index in test_items]

    report_progress(message="Loading the current model and measuring it before retraining")
    model = load_trained(model_service.model_path)
    compile_model(model, learning_rate=learning_rate, class_weights=class_weights)
    scores_before = evaluate_dataset(model, test_dataset, test_labels, class_weights)

    def report_epoch(epoch_row):
        report_progress(message=f"Training: epoch {epoch_row['epoch']}/{epochs}", epoch=epoch_row["epoch"],
                        total_epochs=epochs, accuracy=epoch_row["accuracy"],
                        val_accuracy=epoch_row["val_accuracy"], val_precision=epoch_row["val_precision"],
                        val_recall=epoch_row["val_recall"], val_f1_score=epoch_row["val_f1_score"],
                        val_roc_auc=epoch_row["val_roc_auc"])

    epoch_metrics = EpochMetrics(training_eval_dataset, training_labels, validation_dataset, validation_labels,
                                 class_weights, on_epoch_finished=report_epoch, print_each_epoch=False)
    report_progress(message="Training", epoch=0, total_epochs=epochs)
    model.fit(training_dataset, validation_data=validation_dataset, epochs=epochs, verbose=0, shuffle=False,
              callbacks=[epoch_metrics,
                         tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=RETRAIN_PATIENCE,
                                                          restore_best_weights=True)])

    report_progress(message="Evaluating the retrained model")
    scores_after = evaluate_dataset(model, test_dataset, test_labels, class_weights)

    new_version_path = os.path.join(models_folder,
                                    f"tomato_mobilenetv2_v{len(database.model_history()) + 1}.keras")
    model.save(new_version_path)
    database.add_model_version(new_version_path, scores_after, n_new=len(pending_uploads), make_current=True)
    database.mark_used([upload["id"] for upload in pending_uploads])
    model_service.load(new_version_path)
    return {"before": scores_before, "after": scores_after, "new_images": len(pending_uploads),
            "model_path": new_version_path,
            "class_weights": dict(zip(class_names, class_weights)), "epochs": epoch_metrics.history_rows}
