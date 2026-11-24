# Tomato Leaf Disease: End-to-End ML Pipeline

A custom CNN that classifies tomato leaf photos into **Early blight, Late blight, Leaf mold or healthy**, wrapped in a
FastAPI service and web UI with prediction, data insights, bulk upload, a database and one-click retraining.
Load-tested with Locust against 1, 2 and 4 Docker containers.

| | |
|---|---|
| **Demo video (YouTube)** | TODO: paste link |
| **Live URL** | TODO: paste deployed URL |
| **Dataset** | PlantVillage tomato leaves (Kaggle), 4 classes, images (not tabular) |

## Project description
- **Task:** 4-class image classification of tomato leaves.
- **Model:** custom CNN (4 conv blocks, He init, L2, dropout, Adam), no pre-trained weights.
- **Optimization:** learning rate and dropout search, early stopping, ReduceLROnPlateau, data augmentation.
- **Result on the held-out test set (708 train / 177 test images per class in the full run):**

| Model | Accuracy | Precision | Recall | F1 | Loss |
|---|---|---|---|---|---|
| Baseline CNN | 80.8% | 0.843 | 0.808 | 0.803 | 0.567 |
| Optimized CNN | 88.8% | 0.895 | 0.888 | 0.887 | 0.388 |

The weakest class is Late blight (recall 76.3%), mostly confused with Early blight. Details and plots are in the notebook.

## Features
1. **Prediction:** upload one leaf photo, get the class and probabilities.
2. **Visualizations:** class balance, average colour, brightness and texture per class, each with a written interpretation.
3. **Bulk upload:** upload many labelled photos; each is validated, stored on disk and recorded in a SQLite database.
4. **Retraining:** a button fine-tunes the current model on the pending uploads (database -> preprocessing -> training -> evaluation),
   saves a new model version, records its metrics and hot-swaps it into the running API.
5. **Uptime:** the UI shows how long the service has been running, the model version and the container name.

## Repository structure
```
README.md
notebook/tomato_leaf_disease.ipynb   data exploration, preprocessing, model training, evaluation (run on Google Colab, GPU)
src/preprocessing.py                 image preprocessing, dataset loading, augmentation, dataset insights
src/model.py                         CNN definition, loading, evaluation metrics
src/prediction.py                    prediction service
src/retrain.py                       retraining pipeline
src/database.py                      SQLite: uploaded images and model versions
app/api.py                           FastAPI backend
app/static/index.html                web UI
data/train, data/test                sample of the data (60 train / 20 test images per class), plus data/holdout for demos
models/tomato_cnn.h5                 trained model; class_names.json; metrics.json
locust/locustfile.py                 load test
scripts/run_locust_experiments.sh    runs Locust with 1, 2 and 4 containers
Dockerfile, docker-compose.yml, nginx.conf
```
The repository holds a small **sample** of the dataset to keep it light; the notebook documents the full data split.
The notebook contains Colab-specific cells (Drive mount, Kaggle download); its saved outputs are the results above.

## Setup

### Option A: run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.api:app --port 8000
```
Open http://localhost:8000

### Option B: Docker
```bash
docker build -t tomato-app .
docker run -p 8000:8000 tomato-app
```
Open http://localhost:8000

### Option C: Docker Compose with a load balancer (used for the Locust tests)
```bash
docker compose up -d --build --scale api=2     # any number of containers
docker compose restart nginx
```
Open http://localhost:8080

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status, uptime, model version, container name |
| POST | `/api/predict` | form field `file` (image) -> label, confidence, probabilities |
| POST | `/api/upload` | form fields `label` and `files` (many images) -> saved to disk and database |
| POST | `/api/retrain` | start retraining on pending uploads |
| GET | `/api/retrain/status` | progress and before/after metrics |
| GET | `/api/insights` | data behind the visualizations |
| GET | `/api/model` | current model and version history |

## Load testing with Locust
```bash
pip install -r requirements-dev.txt
./scripts/run_locust_experiments.sh        # 50 users, 90 s, for 1, 2 and 4 containers; CSVs go to results/
```
Results: TODO fill in from `results/containers_*_stats.csv` (50 users, 5 users/s spawn rate, 90 s).

| Containers | Requests | Requests/s | Median latency (ms) | 95th percentile (ms) | Failures |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 4 | | | | | |

## Notes
- Uploaded images and the database live on the container's disk. With several containers each one has its own copy, so retraining is
  meant for a single instance. On cloud platforms with a temporary disk they reset on redeploy.
- TensorFlow needs roughly 1 GB of RAM or more; choose a cloud instance size accordingly.
