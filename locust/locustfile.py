"""Flood test: users upload a leaf photo for prediction and sometimes check the health endpoint."""
import os

from locust import HttpUser, between, task

SAMPLE = os.path.join(os.path.dirname(__file__), "sample.jpg")


class LeafUser(HttpUser):
    wait_time = between(0.5, 2)

    def on_start(self):
        with open(SAMPLE, "rb") as f:
            self.image = f.read()

    @task(5)
    def predict(self):
        self.client.post("/api/predict", files={"file": ("leaf.jpg", self.image, "image/jpeg")})

    @task(1)
    def health(self):
        self.client.get("/api/health")
