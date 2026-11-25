"""Preprocessing shared by the training notebook, the web app and the retraining code.

Training and serving use the SAME TensorFlow operations (decode -> float -> bilinear resize to 224 x 224),
so a photo gives the same model input whether it comes from a training folder or an upload.
Pixels stay in the 0-255 range here; the first layer of the model maps them to [-1, 1] for MobileNetV2.

The train/validation split is decided by a hash of each file's relative path, not by a random shuffle.
A file therefore always lands in the same subset, and new uploads never move existing images.
"""
import hashlib
import io
import os

import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)                  # MobileNetV2's native input size
INSIGHT_SIZE = (128, 128)              # size used only for the dashboard colour and brightness statistics
BATCH_SIZE = 32
VALIDATION_FRACTION = 0.2
RANDOM_SEED = 42
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
AUTOTUNE = tf.data.AUTOTUNE


# ---------- decoding (shared by training and serving) ----------

def decode_and_resize(image_bytes, image_size=IMG_SIZE):
    """Decode image bytes (a scalar string tensor) and return a float32 tensor of shape (H, W, 3), values 0-255."""
    decoded = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    resized = tf.image.resize(tf.cast(decoded, tf.float32), image_size, method="bilinear")
    resized.set_shape((image_size[0], image_size[1], 3))
    return resized


def preprocess_bytes(image_bytes):
    """Prepare ONE photo given as bytes for prediction. Returns a float32 array of shape (1, 224, 224, 3)."""
    try:
        model_input = decode_and_resize(tf.constant(image_bytes))
    except tf.errors.InvalidArgumentError:
        # a format TensorFlow cannot decode (for example TIFF): decode with Pillow and re-encode losslessly as PNG
        buffer = io.BytesIO()
        Image.open(io.BytesIO(image_bytes)).convert("RGB").save(buffer, "PNG")
        model_input = decode_and_resize(tf.constant(buffer.getvalue()))
    return model_input.numpy()[None, ...]


def preprocess_path(image_path):
    with open(image_path, "rb") as image_file:
        return preprocess_bytes(image_file.read())


# ---------- file lists and the train / validation split ----------

def list_image_files(folder):
    return sorted(os.path.join(folder, name) for name in os.listdir(folder) if name.lower().endswith(IMAGE_EXTENSIONS))


def labelled_items(root_folder, class_names):
    """Return (image_path, class_index) for every photo in root_folder/<class_name>/. Missing class folders are skipped."""
    items = []
    for class_index, class_name in enumerate(class_names):
        class_folder = os.path.join(root_folder, class_name)
        if os.path.isdir(class_folder):
            items.extend((image_path, class_index) for image_path in list_image_files(class_folder))
    return items


def _hash_bucket(text):
    """A number between 0 and 1 that depends only on the text, so the same file always gets the same number."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_train_validation(items, root_folder, validation_fraction=VALIDATION_FRACTION):
    """Deterministic train/validation split. The key is the path relative to root_folder, so it never changes."""
    training_items, validation_items = [], []
    for image_path, class_index in items:
        relative_path = os.path.relpath(image_path, root_folder).replace("\\", "/")
        target = validation_items if _hash_bucket(relative_path) < validation_fraction else training_items
        target.append((image_path, class_index))
    return training_items, validation_items


def count_per_class(items, number_of_classes):
    counts = np.zeros(number_of_classes, dtype=int)
    for _, class_index in items:
        counts[class_index] += 1
    return counts


# ---------- tf.data pipelines ----------

_augmentation_layers = None


def get_augmentation():
    """Random flips, rotation, zoom and brightness, applied to training photos only."""
    global _augmentation_layers
    if _augmentation_layers is None:
        from tensorflow.keras import layers
        _augmentation_layers = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.2, value_range=(0, 255)),
        ], name="augmentation")
    return _augmentation_layers


def make_dataset(items, number_of_classes, shuffle=False, augment=False, batch_size=BATCH_SIZE, seed=RANDOM_SEED):
    """Batches of (images, one-hot labels). The order is kept unless shuffle=True."""
    image_paths = [image_path for image_path, _ in items]
    class_indices = [class_index for _, class_index in items]
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, class_indices))
    if shuffle:
        dataset = dataset.shuffle(len(items), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda image_path, class_index: (decode_and_resize(tf.io.read_file(image_path)),
                                         tf.one_hot(class_index, number_of_classes)),
        num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    if augment:
        augmentation = get_augmentation()
        dataset = dataset.map(lambda images, labels: (augmentation(images, training=True), labels),
                              num_parallel_calls=AUTOTUNE)
    return dataset.prefetch(AUTOTUNE)


# ---------- dashboard statistics (web app only) ----------
# The functions below feed the dashboard charts of the web app. Training does not use them.

def image_features(image_path):
    """Mean red, green, blue, brightness and texture (spread of grey values) of one photo."""
    image = Image.open(image_path).convert("RGB").resize(INSIGHT_SIZE)
    colour_values = np.array(image, dtype=np.float32)
    grey_values = np.array(image.convert("L"), dtype=np.float32)
    return [float(colour_values[:, :, 0].mean()), float(colour_values[:, :, 1].mean()),
            float(colour_values[:, :, 2].mean()), float(grey_values.mean()), float(grey_values.std())]


def dataset_insights(train_dir, class_names, sample_size_per_class=60, seed=RANDOM_SEED):
    """Per class: photo count, and the median colour, brightness and texture of a fixed random sample."""
    import random
    rng = random.Random(seed)
    insights = {"classes": [], "count": [], "mean_r": [], "mean_g": [], "mean_b": [],
                "brightness": [], "texture": [], "sample_size": sample_size_per_class}
    for class_name in class_names:
        class_folder = os.path.join(train_dir, class_name)
        photo_paths = list_image_files(class_folder) if os.path.isdir(class_folder) else []
        sample = rng.sample(photo_paths, min(sample_size_per_class, len(photo_paths)))
        features = np.array([image_features(p) for p in sample]) if sample else np.zeros((1, 5))
        medians = np.median(features, axis=0)
        insights["classes"].append(class_name)
        insights["count"].append(len(photo_paths))
        for key, value in zip(["mean_r", "mean_g", "mean_b", "brightness", "texture"], medians):
            insights[key].append(round(float(value), 1))
    return insights
