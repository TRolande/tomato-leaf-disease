"""FastAPI backend: prediction, bulk upload with uploader name, retraining, dataset insights and health.

Run from the repository root:  uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 1
One worker only: the model, the SQLite database and the retraining thread live in this process.
"""
import json
import logging
import socket
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.database import Database
from src.prediction import ModelService
from src.preprocessing import dataset_insights
from src.retrain import retrain
from src.uploads import MAX_BYTES, UploadError, read_photo, save_upload, validate_username

log = logging.getLogger("tomato-api")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
MODELS_DIR = ROOT / "models"
BASE_MODEL = MODELS_DIR / "tomato_mobilenetv2.keras"
CLASS_NAMES = MODELS_DIR / "class_names.json"
BASE_METRICS = MODELS_DIR / "metrics.json"
RETRAIN_EPOCHS = 8

START_TIME = time.time()


class State:
    db = None
    service = None
    insights = None


state = State()
_retrain_lock = threading.Lock()


def _resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _start_service(db):
    """Load the current model from the database. If it is missing or was built for other preprocessing
    (for example an older 128 px model), fall back to the base model shipped in models/."""
    current = db.current_model()
    if current and _resolve(current["path"]).exists():
        try:
            return ModelService(str(_resolve(current["path"])), str(CLASS_NAMES))
        except ValueError as e:
            log.warning("current model not usable, falling back to the base model: %s", e)
    metrics = json.loads(BASE_METRICS.read_text())
    db.add_model_version(str(BASE_MODEL.relative_to(ROOT)), metrics, n_new=0, make_current=True)
    return ModelService(str(BASE_MODEL), str(CLASS_NAMES))


@asynccontextmanager
async def lifespan(app):
    state.db = Database(str(DATA_DIR / "app.db"))
    state.db.mark_interrupted_retrain_runs()     # a run left "running" by a stopped server cannot be continued
    state.service = _start_service(state.db)
    yield


app = FastAPI(title="Tomato Leaf Disease API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(ROOT / "app" / "static")), name="static")


PAGES = ["index.html", "dataset.html", "upload.html", "retrain.html"]


def _page(name):
    return FileResponse(ROOT / "app" / "static" / name)


@app.get("/")
def index():
    return _page("index.html")


for _name in PAGES:                      # every page also has its own address, e.g. /dataset.html (the nav links use these)
    app.add_api_route(f"/{_name}", (lambda n: (lambda: _page(n)))(_name), methods=["GET"], include_in_schema=False)


@app.get("/api/health")
def health():
    current = state.db.current_model()
    return {"status": "ok", "uptime_seconds": round(time.time() - START_TIME, 1),
            "model_version": current["id"], "model_accuracy": current["accuracy"],
            "container": socket.gethostname()}


@app.get("/api/model")
def model_info():
    return {"current": state.db.current_model(), "history": state.db.model_history(),
            "classes": state.service.class_names, "input_size": list(state.service.model.input_shape[1:3])}


@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"{file.filename}: file is larger than 10 MB")
    try:
        read_photo(data, file.filename or "photo")          # Pillow must be able to open it
    except UploadError as e:
        raise HTTPException(400, str(e))
    return state.service.predict(data)


@app.post("/api/upload")
def upload(label: str = Form(...), username: str = Form(...), files: List[UploadFile] = File(...)):
    """Bulk upload. Each photo is validated, recorded in the database with its uploader, and saved to disk
    with the username in its EXIF metadata. Photos that fail are reported back, the rest are still saved."""
    try:
        username = validate_username(username)
    except UploadError as e:
        raise HTTPException(400, str(e))
    if label not in state.service.class_names:
        raise HTTPException(400, f"Unknown class '{label}'. Choose one of {state.service.class_names}")

    saved, rejected = [], []
    for f in files:
        try:
            data = f.file.read(MAX_BYTES + 1)
            saved.append(save_upload(state.db, data, f.filename, label, username,
                                     state.service.class_names, UPLOAD_DIR))
        except UploadError as e:
            rejected.append({"file": f.filename, "reason": str(e)})
        except Exception:
            log.exception("could not store upload %s", f.filename)
            rejected.append({"file": f.filename, "reason": "could not be stored, try again"})
    return {"saved": len(saved), "uploads": [{k: s[k] for k in ("id", "filename", "label", "username")} for s in saved],
            "rejected": rejected, "counts": state.db.upload_counts()}


@app.get("/api/uploads")
def uploads():
    return {"counts": state.db.upload_counts(), "pending": len(state.db.pending_uploads()),
            "recent": state.db.recent_uploads(20)}


def _run_retrain(run_id):
    """Background thread: runs retrain() and writes each step to the database, so a reload can show it."""
    database = state.db

    def on_progress(**details):
        changes = {key: details[key] for key in ("message", "epoch", "total_epochs") if key in details}
        if changes:
            database.update_retrain_run(run_id, **changes)
        if details.get("epoch"):
            database.add_retrain_epoch(run_id, details["epoch"], details)

    try:
        result = retrain(database, state.service, str(TRAIN_DIR), str(TEST_DIR), str(MODELS_DIR),
                         epochs=RETRAIN_EPOCHS, on_progress=on_progress)
        state.insights = None                       # training data changed, redraw the insights
        database.update_retrain_run(run_id, status="done", finished_at=time.time(),
                                    message="Retraining finished. New model is live.",
                                    result_json=json.dumps({k: result[k] for k in ("before", "after", "new_images")}))
    except Exception as error:
        log.exception("retraining failed")
        database.update_retrain_run(run_id, status="error", finished_at=time.time(),
                                    message=str(error) or "Retraining failed.")


def _status_payload(run):
    if run is None:
        return {"status": "idle", "message": "No retraining run yet.", "epochs": []}
    last_epoch = run["epochs"][-1] if run["epochs"] else {}
    payload = {"run_id": run["id"], "status": run["status"], "message": run["message"] or "",
               "epoch": run["epoch"], "total_epochs": run["total_epochs"], "started_at": run["started_at"],
               "finished_at": run["finished_at"], "epochs": run["epochs"],
               "val_accuracy": last_epoch.get("val_accuracy")}
    if run["status"] == "done" and run["result"]:
        payload["result"] = run["result"]
    return payload


@app.post("/api/retrain")
def start_retrain():
    with _retrain_lock:
        if state.db.running_retrain_run() is not None:
            raise HTTPException(409, "Retraining is already running.")
        if not state.db.pending_uploads():
            raise HTTPException(400, "No new uploaded photos. Upload photos first.")
        run_id = state.db.start_retrain_run(RETRAIN_EPOCHS)
    threading.Thread(target=_run_retrain, args=(run_id,), daemon=True).start()
    return _status_payload(state.db.latest_retrain_run())


@app.get("/api/retrain/status")
def retrain_status():
    return _status_payload(state.db.latest_retrain_run())


@app.get("/api/insights")
def insights():
    if state.insights is None:
        state.insights = dataset_insights(str(TRAIN_DIR), state.service.class_names)
    return state.insights
