"""SQLite storage: uploaded photos, model versions, and every retraining run with its epochs."""
import json
import os
import sqlite3
import time
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,                  -- the name the user gave the file
    stored_path TEXT NOT NULL,               -- where the re-encoded JPEG lives on disk
    label TEXT NOT NULL,
    username TEXT,                           -- who uploaded it (also written into the image EXIF)
    sha256 TEXT,                             -- hash of the stored file, to detect changes or duplicates
    uploaded_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'   -- pending = not trained on yet, used = included in a retraining run
);
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    accuracy REAL, loss REAL, precision REAL, recall REAL, f1 REAL, roc_auc REAL,
    n_new_images INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS retrain_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL,                    -- running, done, error or interrupted
    message TEXT,
    epoch INTEGER NOT NULL DEFAULT 0,
    total_epochs INTEGER NOT NULL DEFAULT 0,
    started_at REAL NOT NULL,
    finished_at REAL,
    result_json TEXT                         -- before/after metrics, set when the run is done
);
CREATE TABLE IF NOT EXISTS retrain_epochs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    values_json TEXT NOT NULL,               -- the metrics reported after this epoch
    UNIQUE (run_id, epoch)
);
"""

# columns added after the first release; ALTER TABLE brings an existing app.db up to date
MIGRATIONS = {
    "uploads": {"username": "TEXT", "sha256": "TEXT"},
    "model_versions": {"roc_auc": "REAL"},
}


class Database:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)
            self._migrate(c)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    @staticmethod
    def _migrate(c):
        for table, columns in MIGRATIONS.items():
            existing = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
            for name, kind in columns.items():
                if name not in existing:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")

    # ----- uploads -----
    def add_upload(self, filename, stored_path, label, username, sha256=None):
        with self.conn() as c:
            cur = c.execute(
                "INSERT INTO uploads (filename, stored_path, label, username, sha256, uploaded_at) VALUES (?,?,?,?,?,?)",
                (filename, stored_path, label, username, sha256, time.time()))
            return cur.lastrowid

    def set_upload_hash(self, upload_id, sha256):
        with self.conn() as c:
            c.execute("UPDATE uploads SET sha256=? WHERE id=?", (sha256, upload_id))

    def delete_upload(self, upload_id):
        with self.conn() as c:
            c.execute("DELETE FROM uploads WHERE id=?", (upload_id,))

    def get_upload(self, upload_id):
        with self.conn() as c:
            r = c.execute("SELECT * FROM uploads WHERE id=?", (upload_id,)).fetchone()
        return dict(r) if r else None

    def pending_uploads(self):
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM uploads WHERE status='pending' ORDER BY id")]

    def recent_uploads(self, limit=20):
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT id, filename, label, username, status, uploaded_at "
                                               "FROM uploads ORDER BY id DESC LIMIT ?", (limit,))]

    def mark_used(self, ids):
        with self.conn() as c:
            c.executemany("UPDATE uploads SET status='used' WHERE id=?", [(i,) for i in ids])

    def upload_counts(self):
        with self.conn() as c:
            rows = c.execute("SELECT label, status, COUNT(*) n FROM uploads GROUP BY label, status").fetchall()
        out = {}
        for r in rows:
            out.setdefault(r["label"], {"pending": 0, "used": 0})[r["status"]] = r["n"]
        return out

    # ----- model versions -----
    def add_model_version(self, path, metrics, n_new=0, make_current=True):
        with self.conn() as c:
            if make_current:
                c.execute("UPDATE model_versions SET is_current=0")
            cur = c.execute(
                "INSERT INTO model_versions (path, accuracy, loss, precision, recall, f1, roc_auc, n_new_images, created_at, is_current)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (path, metrics.get("accuracy"), metrics.get("loss"), metrics.get("precision"),
                 metrics.get("recall"), metrics.get("f1"), metrics.get("roc_auc"),
                 n_new, time.time(), 1 if make_current else 0))
            return cur.lastrowid

    def current_model(self):
        with self.conn() as c:
            r = c.execute("SELECT * FROM model_versions WHERE is_current=1 ORDER BY id DESC LIMIT 1").fetchone()
        return dict(r) if r else None

    def model_history(self):
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM model_versions ORDER BY id")]

    # ----- retraining runs -----
    def start_retrain_run(self, total_epochs, message="Starting"):
        with self.conn() as c:
            cur = c.execute("INSERT INTO retrain_runs (status, message, total_epochs, started_at) VALUES ('running',?,?,?)",
                            (message, total_epochs, time.time()))
            return cur.lastrowid

    def update_retrain_run(self, run_id, **fields):
        allowed = {"status", "message", "epoch", "total_epochs", "finished_at", "result_json"}
        changes = {k: v for k, v in fields.items() if k in allowed}
        if not changes:
            return
        assignments = ", ".join(f"{name}=?" for name in changes)
        with self.conn() as c:
            c.execute(f"UPDATE retrain_runs SET {assignments} WHERE id=?", (*changes.values(), run_id))

    def add_retrain_epoch(self, run_id, epoch, values):
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO retrain_epochs (run_id, epoch, values_json) VALUES (?,?,?)",
                      (run_id, epoch, json.dumps(values, default=float)))

    def running_retrain_run(self):
        with self.conn() as c:
            r = c.execute("SELECT * FROM retrain_runs WHERE status='running' ORDER BY id DESC LIMIT 1").fetchone()
        return dict(r) if r else None

    def mark_interrupted_retrain_runs(self):
        """Called at startup: a run still marked running belonged to a process that has stopped."""
        with self.conn() as c:
            c.execute("UPDATE retrain_runs SET status='interrupted', finished_at=?, "
                      "message='The server stopped during this run. The photos are still waiting, so you can retrain again.' "
                      "WHERE status='running'", (time.time(),))

    def latest_retrain_run(self):
        with self.conn() as c:
            r = c.execute("SELECT * FROM retrain_runs ORDER BY id DESC LIMIT 1").fetchone()
            if r is None:
                return None
            run = dict(r)
            epochs = c.execute("SELECT values_json FROM retrain_epochs WHERE run_id=? ORDER BY epoch", (run["id"],)).fetchall()
        run["epochs"] = [json.loads(e["values_json"]) for e in epochs]
        run["result"] = json.loads(run["result_json"]) if run["result_json"] else None
        return run
