# Tomato Leaf Check

A web app that looks at a photo of a tomato leaf and says which of four conditions it shows: **Early blight, Late blight, Leaf mold or Healthy**. It also lets you add labelled photos, retrain the model on them, and see how each model version performs.

| | |
|---|---|
| **Demo video** | https://youtu.be/HIDdQSXobRo |
| **Live URL** | https://tomatoleaf.duckdns.org |
| **Dataset** | PlantVillage tomato leaves (Kaggle), colour version, four classes |

## What the project does

- **Check a photo:** the model returns the most likely class, the confidence, and the score for every class. Results below 70 % confidence are flagged.
- **Dataset:** training photo counts, average colour, brightness and texture per class, with a caption for each chart.
- **Upload:** labelled photos with the uploader's username. Each photo is validated, stored on disk and recorded in the database. The username and upload ID are also written into the photo's EXIF metadata.
- **Retrain:** the waiting photos are added to the training data, and the classifier head is fine-tuned on it. The new version replaces the live model when the run finishes. Progress, the per-epoch validation chart and the before/after results are kept in the database.
- **Status:** the header shows whether the server is online, the uptime, the live model version and its accuracy, and the container name.

## Model

- **Base:** MobileNetV2 pretrained on ImageNet, with its weights frozen. Only the classifier head is trained.
- **Head:** global average pooling, dropout, a dense layer of 128 units with L2 weight decay, dropout, and a softmax over the four classes.
- **Input:** 224 × 224 RGB photos. Training and the web app use the same decoding and resizing code (`src/preprocessing.py`).
- **Loss:** class-weighted categorical cross-entropy. The weights come from the class counts and are recomputed on each retraining run.
- **Training settings** (from the notebook): up to 30 epochs, Adam with learning rate 0.001, dropout 0.3. Early stopping watches validation loss (min_delta 0.002, patience 5) and restores the best weights. The learning rate is halved after 3 epochs without improvement. Each epoch reports accuracy, precision, recall, F1 score and ROC AUC for the training and validation photos.
- **Current test result:** 94.31 % accuracy (`models/metrics.json`). The notebook has the full per-class results, the confusion matrices and the Grad-CAM examples.

## Project structure

```
notebook/tomato.ipynb   training, evaluation, per-class results and Grad-CAM
src/preprocessing.py                 decoding and resizing, train/validation split, data pipelines, dataset statistics
src/model.py                         model, class weights, weighted loss, metrics, per-epoch reporting
src/prediction.py                    loads the live model and predicts one photo
src/retrain.py                       retraining run used by the web app
src/uploads.py                       upload validation, EXIF metadata and file hash
src/database.py                      SQLite tables: uploads, model versions, retraining runs and their epochs
app/api.py                           FastAPI backend and page routes
app/static/                          the four pages (index, dataset, upload, retrain), styles.css and their scripts
models/                              tomato_mobilenetv2.keras, class_names.json, class_weights.json, metrics.json
data/train, data/test                sample photos for the dashboard and for retraining
Dockerfile, .dockerignore            API container image
docker-compose.prod.yml              production stack: Caddy (HTTPS) -> nginx (pages and /api) -> API
Caddyfile                            HTTPS and reverse proxy settings
deploy/                              nginx configuration and its image
docker-compose.yml, nginx.conf       local stack with two API containers
locust/, scripts/                    load test files (Locust) and the script that runs it
kaggle/, scripts/push_to_kaggle.sh   Kaggle kernel settings and the script that publishes the notebook
requirements.txt                     runtime packages
requirements-dev.txt                 adds Locust
```

`data/app.db` (the database) and `data/uploads/` are created at runtime and are not in Git.

## Running it

### 1. Get the model files

Run the notebook (Colab, Kaggle or your own machine). Copy `tomato_mobilenetv2.keras`, `class_names.json`, `class_weights.json` and `metrics.json` into `models/`. The app will not start without `tomato_mobilenetv2.keras`.

### 2. Run locally

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Open http://localhost:8000. Use `--workers 1`: the model, the database and the retraining thread all run in that one process.

### 3. Run in Docker

```powershell
docker build --load -t tomato-app .
docker run -d --name tomato-app -p 8080:8000 tomato-app
```

Open http://127.0.0.1:8080. The container keeps its data inside it, so `docker rm` deletes uploads, database and retrained versions. For data that persists, use the production stack.

### 4. Run the production stack

Create `.env` in the project folder:

```
DOMAIN=localhost
```

Then start it:

```powershell
docker compose -f docker-compose.prod.yml up -d --build
```

Open https://localhost and accept the certificate warning (the certificate is local). For a server, set `DOMAIN` to a hostname that points at it. Caddy obtains the certificate automatically, and ports 80 and 443 must be open. Data is stored in the named volumes `app_data` and `app_models`, so it survives restarts and rebuilds. Keep the `caddy_data` volume, or Let's Encrypt rate limits apply.

### 5. Deploy on AWS

The live app runs on an AWS EC2 instance (Ubuntu 26.04, eu-north-1). It uses an Elastic IP and a free DuckDNS name, so it needs no paid domain.

1. Launch an instance with a key pair. Open ports 22 (SSH), 80 and 443 in its security group.
2. Allocate an **Elastic IP** and associate it with the instance (here 13.61.54.133).
3. On duckdns.org, create a name (here `tomatoleaf.duckdns.org`) and set its IP to the Elastic IP.
4. Connect to the instance, then run:
   ```bash
   sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
   sudo usermod -aG docker ubuntu
   git clone https://github.com/TRolande/tomato-leaf-disease.git
   cd tomato-leaf-disease
   echo "DOMAIN=tomatoleaf.duckdns.org" > .env
   sudo docker compose -f docker-compose.prod.yml up -d --build
   ```
5. Caddy gets the HTTPS certificate for the name automatically. Check its log with `sudo docker compose -f docker-compose.prod.yml logs caddy`, and look for a line saying the certificate was obtained.

Keep the instance running, because the public link only works while it is. The public address is https://tomatoleaf.duckdns.org. Browsers show the connection as secure, and the certificate is valid.

## Pages and API

| Page | Address | What it does |
|---|---|---|
| Check a photo | `/` | Predicts the class of one photo |
| Dataset | `/dataset.html` | Statistics and charts of the training photos |
| Upload photos | `/upload.html` | Uploads labelled photos with a username |
| Retrain | `/retrain.html` | Starts retraining, shows progress, results and model versions |

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status, uptime, live model version and accuracy, container name |
| POST | `/api/predict` | form field `file` → label, confidence, probability per class |
| POST | `/api/upload` | form fields `label`, `username`, `files` (repeat for each photo) |
| GET | `/api/uploads` | waiting and used counts per class, and the 20 most recent uploads |
| GET | `/api/model` | live model, version history, class names and input size |
| POST | `/api/retrain` | starts a retraining run (409 if one is running, 400 if no photos are waiting) |
| GET | `/api/retrain/status` | the latest run: status, epoch, message, saved epochs and result |
| GET | `/api/insights` | data behind the dataset charts |

Usernames are 2 to 32 characters: letters, digits, `.`, `_` and `-`. Photos must be JPG or PNG and at most 10 MB.

**Run states:** `idle`, `running`, `done`, `error`, or `interrupted`. If the server stops during a run, the run is marked `interrupted` when the server starts again, and its photos stay waiting for the next run.

## Notes and limits

- **Retraining is slow on CPU.** A full run on the training folder takes a long time, because each epoch predicts every training photo for the metrics as well as training on it. Plan the demo around this, or run it ahead of time.
- **One process.** Run one API worker. The model, the database and the retraining thread all live in that process, and SQLite is not shared across containers. Keep one `api` container.
- **Uploaded photos carry the username.** It is written into each JPEG's EXIF metadata, so keep that in mind when sharing the files.
- The notebook's data split is by photo. The notebook's duplicate check reports how many test photos have a near copy in the training set; check that number before quoting the test results.

## Flood request simulation (Locust)

The app was tested with Locust. Each simulated user uploads a leaf photo to `/api/predict` and checks `/api/health`, with a short pause between requests. Two runs were made, with the same script.

| Run | Server | Users (new per second) | Duration | Requests | Failures | Prediction median | Prediction 95th percentile | Requests per second (all) |
|---|---|---|---|---|---|---|---|---|
| Live site | AWS EC2, 2 vCPUs, one API process | 50 (5/s) | 90 s | 446 | 395 (88.6 %, `502 Bad Gateway`) | 0.30 s | 51 s | 6.38 |
| Local Docker | this computer, one API process | 10 (2/s) | 90 s | 20 | 0 | 82 s | 86 s | 0.23 |

**What it shows.** A single prediction takes well under a second when the server is not busy. Under load, the requests queue up behind the TensorFlow model, which runs one prediction at a time per process. The live server could not keep up with 50 users, and its web server returned `502 Bad Gateway` for most requests. It recovered after a restart. Locally, 10 users already cause queuing, although nothing failed.

**Limits and next steps.** The app is limited by CPU and by running a single process. Handling heavy traffic would need a bigger instance, several API workers with enough memory, or a request queue in front of the model.

The raw Locust output is saved in `results/`, which Git does not track.

## Kaggle

The notebook runs on Colab, on Kaggle, or on your own machine. On Kaggle, attach the `plant-village` dataset, and turn on the GPU and internet. To publish it as a kernel, put your username in `kaggle/kernel-metadata.json` and run `bash scripts/push_to_kaggle.sh`.
