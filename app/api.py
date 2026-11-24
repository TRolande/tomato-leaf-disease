"""FastAPI backend: prediction, bulk upload, retraining, dataset insights and health/uptime.

Run from the repository root:  uvicorn app.api:app --host 0.0.0.0 --port 8000
"""
import io
import json
import socket
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.database import Database
from src.prediction import ModelService
from src.preprocessing import dataset_insights
from src.retrain import retrain

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
MODELS_DIR = ROOT / "models"
BASE_MODEL = MODELS_DIR / "tomato_cnn.h5"
CLASS_NAMES = MODELS_DIR / "class_names.json"
MAX_BYTES = 10 * 1024 * 1024
RETRAIN_EPOCHS = 8

START_TIME = time.time()


class State:
    db = None
    service = None
    insights = None
    retrain = {"status": "idle", "message": "No retraining run yet."}


state = State()


def _resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


@asynccontextmanager
async def lifespan(app):
    state.db = Database(str(DATA_DIR / "app.db"))
    current = state.db.current_model()
    if current is None or not _resolve(current["path"]).exists():
        metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
        state.db.add_model_version(str(BASE_MODEL.relative_to(ROOT)), metrics, n_new=0, make_current=True)
        current = state.db.current_model()
    state.service = ModelService(str(_resolve(current["path"])), str(CLASS_NAMES))
    yield


app = FastAPI(title="Tomato Leaf Disease API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(ROOT / "app" / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/health")
def health():
    current = state.db.current_model()
    return {"status": "ok", "uptime_seconds": round(time.time() - START_TIME, 1),
            "model_version": current["id"], "model_accuracy": current["accuracy"],
            "container": socket.gethostname()}


@app.get("/api/model")
def model_info():
    return {"current": state.db.current_model(), "history": state.db.model_history(),
            "classes": state.service.class_names}


def _read_image(upload: UploadFile):
    data = upload.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"{upload.filename}: file is larger than 10 MB")
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise HTTPException(400, f"{upload.filename}: not a valid image (use JPG or PNG)")
    return data


@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    data = _read_image(file)
    return state.service.predict(data)


@app.post("/api/upload")
def upload(label: str = Form(...), files: List[UploadFile] = File(...)):
    """Bulk upload: every file is validated, stored on disk and recorded in the database as 'pending'."""
    if label not in state.service.class_names:
        raise HTTPException(400, f"Unknown class '{label}'. Choose one of {state.service.class_names}")
    saved, rejected = 0, []
    folder = UPLOAD_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    for f in files:
        try:
            data = f.file.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise ValueError("larger than 10 MB")
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            rejected.append({"file": f.filename, "reason": str(e) or "not a valid image"})
            continue
        dest = folder / f"{uuid.uuid4().hex[:10]}.jpg"
        img.save(dest, "JPEG", quality=95)          # one format for every upload
        state.db.add_upload(f.filename or dest.name, str(dest), label)
        saved += 1
    return {"saved": saved, "rejected": rejected, "counts": state.db.upload_counts()}


@app.get("/api/uploads")
def uploads():
    return {"counts": state.db.upload_counts(), "pending": len(state.db.pending_uploads())}


def _progress(**kw):
    state.retrain.update(kw)


def _run_retrain():
    try:
        result = retrain(state.db, state.service, str(TRAIN_DIR), str(TEST_DIR), str(MODELS_DIR),
                         epochs=RETRAIN_EPOCHS, on_progress=_progress)
        state.insights = None                       # training data changed, redraw the insights
        state.retrain.update(status="done", result=result, message="Retraining finished. New model is live.")
    except Exception as e:
        state.retrain.update(status="error", message=str(e))


@app.post("/api/retrain")
def start_retrain():
    if state.retrain.get("status") == "running":
        raise HTTPException(409, "Retraining is already running.")
    if not state.db.pending_uploads():
        raise HTTPException(400, "No new uploaded images. Upload images first.")
    state.retrain = {"status": "running", "message": "Starting", "epoch": 0, "total_epochs": RETRAIN_EPOCHS,
                     "started_at": time.time()}
    threading.Thread(target=_run_retrain, daemon=True).start()
    return state.retrain


@app.get("/api/retrain/status")
def retrain_status():
    return state.retrain


@app.get("/api/insights")
def insights():
    if state.insights is None:
        state.insights = dataset_insights(str(TRAIN_DIR), state.service.class_names)
    return state.insights
