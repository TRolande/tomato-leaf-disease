"""SQLite storage for uploaded training images and model versions (the retraining database)."""
import os
import sqlite3
import time
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    label TEXT NOT NULL,
    uploaded_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'   -- pending = not trained on yet, used = included in a retraining run
);
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    accuracy REAL, loss REAL, precision REAL, recall REAL, f1 REAL,
    n_new_images INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 0
);
"""


class Database:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # ----- uploads -----
    def add_upload(self, filename, stored_path, label):
        with self.conn() as c:
            cur = c.execute("INSERT INTO uploads (filename, stored_path, label, uploaded_at) VALUES (?,?,?,?)",
                            (filename, stored_path, label, time.time()))
            return cur.lastrowid

    def pending_uploads(self):
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM uploads WHERE status='pending' ORDER BY id")]

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
                "INSERT INTO model_versions (path, accuracy, loss, precision, recall, f1, n_new_images, created_at, is_current)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (path, metrics.get("accuracy"), metrics.get("loss"), metrics.get("precision"),
                 metrics.get("recall"), metrics.get("f1"), n_new, time.time(), 1 if make_current else 0))
            return cur.lastrowid

    def current_model(self):
        with self.conn() as c:
            r = c.execute("SELECT * FROM model_versions WHERE is_current=1 ORDER BY id DESC LIMIT 1").fetchone()
        return dict(r) if r else None

    def model_history(self):
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM model_versions ORDER BY id")]
