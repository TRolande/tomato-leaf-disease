"""Prediction service: holds the current model in memory and predicts one image at a time."""
import json
import threading
import time

import numpy as np

from src.model import load_trained
from src.preprocessing import IMG_SIZE, preprocess_bytes


class ModelService:
    def __init__(self, model_path, class_names_path):
        with open(class_names_path) as f:
            self.class_names = json.load(f)
        self._lock = threading.Lock()
        self.model = None
        self.model_path = None
        self.loaded_at = None
        self.load(model_path)

    def load(self, path):
        """Load (or hot-swap) the model. Predictions keep using the old model until the new one is ready."""
        model = load_trained(path)
        shape = tuple(model.input_shape[1:3])
        if shape != tuple(IMG_SIZE):
            raise ValueError(f"{path} expects {shape} images but the preprocessing produces {IMG_SIZE}")
        if model.output_shape[-1] != len(self.class_names):
            raise ValueError(f"{path} has {model.output_shape[-1]} outputs but there are {len(self.class_names)} classes")
        with self._lock:
            self.model, self.model_path, self.loaded_at = model, path, time.time()

    def predict(self, data):
        """data: the image bytes. Returns the label, its confidence and the probability of every class."""
        x = preprocess_bytes(data)
        with self._lock:
            model = self.model
        probs = model(x, training=False).numpy()[0]
        idx = int(np.argmax(probs))
        return {"label": self.class_names[idx],
                "confidence": float(probs[idx]),
                "probabilities": {n: float(p) for n, p in zip(self.class_names, probs)}}
