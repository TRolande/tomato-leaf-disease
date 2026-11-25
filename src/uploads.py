"""Saving uploaded photos: validation, the database row, and the image file with metadata.

Every accepted photo is:
  1. checked (username, class, size, decodable image),
  2. recorded in the `uploads` table (filename, label, username, path),
  3. re-encoded as JPEG and saved under data/uploads/<label>/<random>.jpg,
     with the username and the upload id written into the EXIF metadata,
  4. hashed (sha256) and the hash stored in the same row.
If step 3 fails, the database row is removed again, so no row points at a missing file.
"""
import hashlib
import io
import re
import uuid
from pathlib import Path

from PIL import Image

MAX_BYTES = 10 * 1024 * 1024
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{2,32}$")
EXIF_ARTIST = 0x013B            # standard EXIF "Artist" tag: the uploader's username
EXIF_DESCRIPTION = 0x010E       # standard EXIF "ImageDescription": upload id, label and username


class UploadError(ValueError):
    """A problem with one upload that should be reported to the user, not as a server error."""


def validate_username(username):
    name = (username or "").strip()
    if not USERNAME_RE.match(name):
        raise UploadError("Username must be 2-32 characters: letters, digits, '.', '_' or '-'.")
    return name


def read_photo(data, filename="photo"):
    """Check the bytes and return an RGB PIL image. Raises UploadError if the file is not a usable photo."""
    if len(data) > MAX_BYTES:
        raise UploadError(f"{filename}: larger than 10 MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except Exception:
        raise UploadError(f"{filename}: not a valid image (use JPG or PNG)")


def save_upload(db, data, filename, label, username, class_names, uploads_dir):
    """Store one uploaded photo. Returns a dict with the database id and the stored path."""
    username = validate_username(username)
    if label not in class_names:
        raise UploadError(f"Unknown class '{label}'. Choose one of {list(class_names)}")
    img = read_photo(data, filename or "photo")

    folder = Path(uploads_dir) / label
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{uuid.uuid4().hex}.jpg"
    upload_id = db.add_upload(filename or dest.name, str(dest), label, username)
    try:
        exif = Image.Exif()
        exif[EXIF_ARTIST] = username
        exif[EXIF_DESCRIPTION] = f"upload_id={upload_id}; label={label}; uploaded_by={username}"
        img.save(dest, "JPEG", quality=95, exif=exif)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        db.set_upload_hash(upload_id, digest)
    except Exception:
        db.delete_upload(upload_id)
        dest.unlink(missing_ok=True)
        raise
    return {"id": upload_id, "filename": filename or dest.name, "label": label,
            "username": username, "stored_path": str(dest), "sha256": digest}


def read_metadata(path):
    """Return the uploader (EXIF Artist) and the description (upload id, label, username) of a saved photo."""
    exif = Image.open(path).getexif()
    description = exif.get(EXIF_DESCRIPTION, "")
    return {"artist": exif.get(EXIF_ARTIST), "description": description}
