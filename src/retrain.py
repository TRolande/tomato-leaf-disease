"""Retraining pipeline: DB -> training folder -> preprocessing -> fine-tune -> evaluate -> new model version."""
import os
import shutil

from src.model import compile_model, evaluate_model, load_trained
from src.preprocessing import load_test, load_train_val, prepare


def retrain(db, service, train_dir, test_dir, models_dir, epochs=8, lr=1e-4, on_progress=None):
    """Fine-tune the current model on all pending uploads stored in the database.

    1. read the pending uploads from the database and add them to the training folder
    2. preprocess with the same pipeline as the first training (resize, augmentation on training only)
    3. continue training the current model with a small learning rate (fine-tuning, not starting over)
    4. evaluate before and after on the untouched test set
    5. save a new model version, store its metrics in the database and hot-swap it into the API
    """
    import tensorflow as tf
    progress = on_progress or (lambda **kw: None)

    pending = db.pending_uploads()
    if not pending:
        raise ValueError("No new uploaded images to retrain on. Upload some images first.")

    progress(message=f"Adding {len(pending)} uploaded images to the training data")
    for row in pending:
        dest_dir = os.path.join(train_dir, row["label"])
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(row["stored_path"], os.path.join(dest_dir, f"new_{row['id']}_{row['filename']}"))

    progress(message="Preprocessing images")
    train_ds, val_ds = load_train_val(train_dir)
    train_ds, val_ds = prepare(train_ds, augment=True), prepare(val_ds)
    test_ds = load_test(test_dir)

    progress(message="Loading the current model and measuring it before retraining")
    model = load_trained(service.model_path)
    compile_model(model, lr)
    before = evaluate_model(model, test_ds)

    class Report(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress(message=f"Training: epoch {epoch + 1}/{epochs}", epoch=epoch + 1,
                     train_acc=float(logs.get("accuracy", 0)), val_acc=float(logs.get("val_accuracy", 0)))

    progress(message="Training", epoch=0, total_epochs=epochs)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
                         Report()])

    progress(message="Evaluating the retrained model")
    after = evaluate_model(model, test_ds)

    version_path = os.path.join(models_dir, f"tomato_cnn_v{len(db.model_history()) + 1}.h5")
    model.save(version_path)
    db.add_model_version(version_path, after, n_new=len(pending), make_current=True)
    db.mark_used([r["id"] for r in pending])
    service.load(version_path)
    return {"before": before, "after": after, "new_images": len(pending), "model_path": version_path}
