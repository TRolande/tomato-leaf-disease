"""Preprocessing shared by the notebook, the API and the retraining script.

Every image goes through the same steps the model saw during training:
force 3 colour channels -> resize to 128x128 (bilinear) -> keep pixels 0-255
(the model rescales to 0-1 itself with its first layer).
TensorFlow is imported lazily so the light functions work without it.
"""
import io
import os
import random

import numpy as np
from PIL import Image

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 42


def preprocess_image(source, img_size=IMG_SIZE):
    """Prepare ONE image (path, bytes or file-like) for prediction. Returns shape (1, 128, 128, 3)."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    img = Image.open(source).convert("RGB")
    img = img.resize(img_size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def load_train_val(train_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, seed=SEED):
    """Read class sub-folders (folder name = label) and return train and validation datasets."""
    import tensorflow as tf
    common = dict(validation_split=val_split, seed=seed, image_size=img_size,
                  batch_size=batch_size, label_mode="categorical")
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(train_dir, subset="validation", **common)
    return train_ds, val_ds


def load_test(test_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Test data is not shuffled so predictions line up with labels."""
    import tensorflow as tf
    return tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=img_size, batch_size=batch_size,
        label_mode="categorical", shuffle=False)


_augmentation = None


def get_augmentation():
    """Random flips, rotation, zoom and brightness (same as the notebook). Built once, on first use."""
    global _augmentation
    if _augmentation is None:
        import tensorflow as tf
        from tensorflow.keras import layers
        _augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.2, value_range=(0, 255)),
        ], name="augmentation")
    return _augmentation


def prepare(ds, augment=False):
    """Augment training data only (never validation or test) and prefetch for speed."""
    import tensorflow as tf
    if augment:
        aug = get_augmentation()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ---------- dataset insights (used by the UI visualizations) ----------

def image_features(path):
    """Mean R, G, B, brightness and texture (spread of grey values) of one image."""
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    gray = np.array(img.convert("L"), dtype=np.float32)
    return [float(arr[:, :, 0].mean()), float(arr[:, :, 1].mean()), float(arr[:, :, 2].mean()),
            float(gray.mean()), float(gray.std())]


def dataset_insights(train_dir, class_names, per_class=60, seed=SEED):
    """Per class: image count, and median colour, brightness and texture of a fixed random sample."""
    rng = random.Random(seed)
    out = {"classes": [], "count": [], "mean_r": [], "mean_g": [], "mean_b": [],
           "brightness": [], "texture": [], "sample_size": per_class}
    for cls in class_names:
        folder = os.path.join(train_dir, cls)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
        sample = rng.sample(files, min(per_class, len(files)))
        feats = np.array([image_features(os.path.join(folder, f)) for f in sample]) if sample else np.zeros((1, 5))
        med = np.median(feats, axis=0)
        out["classes"].append(cls)
        out["count"].append(len(files))
        for key, val in zip(["mean_r", "mean_g", "mean_b", "brightness", "texture"], med):
            out[key].append(round(float(val), 1))
    return out
