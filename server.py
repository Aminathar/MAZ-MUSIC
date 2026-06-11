"""
MAZ — AceStep Music Server
==================================
SETUP:
  pip install fastapi uvicorn httpx websockets

RUN:
  uvicorn server:app --host 0.0.0.0 --port 3000

REQUIRES:
  AceStep running on port 8001:
    cd ACE-Step-1.5 && uv run acestep-api
"""

import io
import os
import re
import sys
import uuid
import math
import shutil
import asyncio
import logging
import httpx
import json
import zipfile
import concurrent.futures
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, List
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Header, Request, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env from the same directory as this file before reading os.environ
load_dotenv(Path(__file__).parent / ".env")

import process_guard
import auth as _auth

# ─── Config ───────────────────────────────────────────────────
ACESTEP_API    = os.environ.get("ACESTEP_API", "http://localhost:8001")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin1234")
MAX_PER_DAY    = int(os.environ.get("MAX_PER_DAY", "10"))
POLL_INTERVAL          = 3.0   # seconds between ACE-Step status polls
POLL_TIMEOUT           = 600   # hard wall-clock deadline in seconds
VOICE_PIPELINE_TIMEOUT = int(os.environ.get("VOICE_PIPELINE_TIMEOUT", "1800"))  # 30 min
VOICE_TRAIN_TIMEOUT    = int(os.environ.get("VOICE_TRAIN_TIMEOUT",    "7200"))  # 2 hours
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")   # free key from aistudio.google.com
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
STATIC_DIR     = Path(__file__).parent / "static"
OUTPUT_DIR     = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DB_PATH = Path(__file__).parent / "maz.db"

VOICE_PROFILES_DIR = OUTPUT_DIR / "voice_profiles"
VOICE_SONGS_DIR    = OUTPUT_DIR / "voice_songs"
VOICE_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
VOICE_SONGS_DIR.mkdir(parents=True, exist_ok=True)
RVC_MODELS_DIR     = VOICE_PROFILES_DIR / "rvc_models"
RVC_MODELS_DIR.mkdir(parents=True, exist_ok=True)
VOICE_PIPELINE        = Path(__file__).parent / "voice_pipeline_worker.py"
VOICE_SECTIONS_WORKER = Path(__file__).parent / "voice_sections_worker.py"
VOICE_SECTIONS_DIR    = OUTPUT_DIR / "voice_sections"
VOICE_SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_STORE_DIR = OUTPUT_DIR / "audio"
AUDIO_STORE_DIR.mkdir(parents=True, exist_ok=True)
VOICE_TRAIN_WORKER  = Path(__file__).parent / "rvc_training_worker.py"
RVC_DATASETS_DIR    = VOICE_PROFILES_DIR / "training_datasets"
RVC_TRAINING_OUTPUT = VOICE_PROFILES_DIR / "trained_models"
RVC_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
RVC_TRAINING_OUTPUT.mkdir(parents=True, exist_ok=True)

ACESTEP_AUDIO_DIR = os.environ.get(
    "ACESTEP_AUDIO_DIR",
    str(Path.home() / "Desktop" / "aimusicgenerator" / "ACE-Step-1.5" / ".cache" / "acestep" / "tmp" / "api_audio")
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("maz")

# Suppress noisy Windows ProactorEventLoop socket warnings (harmless — just
# means a WebSocket client disconnected while we were sending an event).
class _SocketSendFilter(logging.Filter):
    def filter(self, record):
        return "socket.send() raised exception" not in record.getMessage()

for _noisy in ("asyncio", "uvicorn.error"):
    logging.getLogger(_noisy).addFilter(_SocketSendFilter())

# ─── In-memory state ──────────────────────────────────────────
generation_queue: List[dict]  = []
active_job: Optional[dict]    = None
_cancelled_jobs: set          = set()   # job_ids cancelled while active
daily_usage: Dict[str, int]   = defaultdict(int)
total_generated = 0
total_failed    = 0
# Keeps the last 100 finished jobs so HTTP /job/{id} can return status after WS broadcast.
recently_finished: Dict[str, dict] = {}
ws_clients: List[WebSocket]  = []
audio_file_map: Dict[str, Path] = {}
analysis_cache: Dict[str, dict] = {}
voice_training_jobs:  Dict[str, dict] = {}
voice_training_procs: Dict[str, object] = {}

_CACHE_MAX = 500  # max entries for unbounded in-memory caches
_audio_scan_sizes: Dict[str, int] = {}  # path → last-seen size for fs-scan stability check

def _cache_set(d: dict, key: str, value) -> None:
    """Insert key→value into d, evicting the oldest entry when over _CACHE_MAX."""
    d[key] = value
    if len(d) > _CACHE_MAX:
        del d[next(iter(d))]

def _datetime_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not serializable: {type(obj)}")

def _persist_queue() -> None:
    """Atomically rewrite pending_queue table to match current generation_queue."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM pending_queue")
            for i, job in enumerate(generation_queue):
                conn.execute(
                    "INSERT INTO pending_queue (job_id, data, position) VALUES (?,?,?)",
                    (job["job_id"], json.dumps(job, default=_datetime_default), i),
                )
            conn.commit()
    except Exception as e:
        log.warning(f"Queue persist failed: {e}")

def _load_queue_from_db() -> None:
    """Restore generation_queue from pending_queue table on startup."""
    global generation_queue
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT data FROM pending_queue ORDER BY position"
            ).fetchall()
        jobs = []
        for (data,) in rows:
            job = json.loads(data)
            if isinstance(job.get("queued_at"), str):
                job["queued_at"] = datetime.fromisoformat(job["queued_at"])
            job["started_at"] = None
            job["status"] = "queued"
            jobs.append(job)
        generation_queue = jobs
        if jobs:
            log.info(f"Restored {len(jobs)} pending job(s) from queue")
    except Exception as e:
        log.warning(f"Queue restore failed: {e}")

# ─── SQLite Database ───────────────────────────────────────────
def init_db():
    """Create tables if they don't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tracks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id      TEXT UNIQUE NOT NULL,
                prompt      TEXT NOT NULL,
                lyrics      TEXT,
                duration    REAL,
                steps       INTEGER,
                guidance    REAL,
                seed        INTEGER,
                bpm_input   INTEGER,
                key_input   TEXT,
                audio_path  TEXT,
                ip          TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis (
                track_id    INTEGER PRIMARY KEY REFERENCES tracks(id),
                bpm         REAL,
                key         TEXT,
                mood        TEXT,
                energy      REAL,
                bass        REAL,
                mid         REAL,
                treble      REAL,
                analyzed_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS presets (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                params      TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL DEFAULT 'user',
                is_active     INTEGER NOT NULL DEFAULT 1,
                created_at    TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_queue (
                job_id   TEXT PRIMARY KEY,
                data     TEXT NOT NULL,
                position INTEGER NOT NULL DEFAULT 0
            )
        """)
        # Migrations — safe to run on existing DBs
        for _col_sql in [
            "ALTER TABLE tracks ADD COLUMN title TEXT",
            "ALTER TABLE tracks ADD COLUMN voice_sections_path TEXT",
            "ALTER TABLE tracks ADD COLUMN favorited INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE tracks ADD COLUMN user_id INTEGER REFERENCES users(id)",
            "ALTER TABLE presets ADD COLUMN user_id INTEGER REFERENCES users(id)",
        ]:
            try:
                conn.execute(_col_sql)
            except Exception:
                pass  # column already exists
        conn.commit()
    log.info(f"Database initialized: {DB_PATH}")


def _ensure_admin_user():
    """Create the default admin account if no users exist yet."""
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
                ("admin", _auth.hash_password(ADMIN_PASSWORD), "admin"),
            )
            conn.commit()
            log.info("Created default admin account (username: admin, password: ADMIN_PASSWORD env var)")

def save_track_to_db(job: dict, audio_path: str):
    """Save a completed track to the database, copying audio to permanent storage."""
    job_id = job["job_id"]
    # Copy audio from ACE-Step temp dir to permanent store so library tracks always work
    if audio_path:
        src = Path(audio_path)
        if src.exists() and src.stat().st_size > 1000:
            dest = AUDIO_STORE_DIR / f"{job_id}{src.suffix}"
            if not dest.exists():
                try:
                    shutil.copy2(str(src), str(dest))
                    log.info(f"Audio stored permanently: {dest.name}")
                except Exception as e:
                    log.warning(f"Could not copy audio to permanent store: {e}")
            if dest.exists():
                audio_path = str(dest)
                # Keep audio_file_map pointing at the permanent copy so
                # /audio/{job_id} immediately serves the stable file.
                _cache_set(audio_file_map, job_id, dest)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO tracks
                (job_id, prompt, lyrics, duration, steps, guidance, seed,
                 bpm_input, key_input, audio_path, ip, user_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                job_id,
                job["prompt"],
                job.get("lyrics"),
                job.get("duration"),
                job.get("infer_steps"),
                job.get("guidance_scale"),
                job.get("seed_out") or job.get("seed"),
                job.get("bpm"),
                job.get("key"),
                audio_path,
                job.get("ip"),
                job.get("user_id"),
            ))
            conn.commit()
        log.info(f"Track saved to DB: {job_id}")
    except Exception as e:
        log.error(f"DB save error: {e}")

def save_analysis_to_db(job_id: str, analysis: dict):
    """Save librosa analysis results to the database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT id FROM tracks WHERE job_id=?", (job_id,)
            ).fetchone()
            if not row:
                return
            conn.execute("""
                INSERT OR REPLACE INTO analysis
                (track_id, bpm, key, mood, energy, bass, mid, treble)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                row[0],
                analysis.get("bpm"),
                analysis.get("key"),
                analysis.get("mood"),
                analysis.get("energy"),
                analysis.get("spectrum", {}).get("bass"),
                analysis.get("spectrum", {}).get("mid"),
                analysis.get("spectrum", {}).get("treble"),
            ))
            conn.commit()
        log.info(f"Analysis saved to DB for job: {job_id}")
    except Exception as e:
        log.error(f"DB analysis save error: {e}")

# ─── App ──────────────────────────────────────────────────────
app = FastAPI(title="MAZ", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/static",  StaticFiles(directory=STATIC_DIR), name="static")

# ─── Helpers ──────────────────────────────────────────────────
def strip_section_headers(lyrics: str) -> str:
    """Remove lines that are purely section labels like [Verse 1], [Қайырма], [Chorus], etc.
    ACE-Step sometimes vocalises these as words instead of treating them as structure."""
    import re
    # Remove section header lines
    cleaned = re.sub(r'(?m)^\s*\[.*?\]\s*$', '', lyrics)
    # Collapse consecutive blank lines into one
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

# ─── Auth dependency ──────────────────────────────────────────
_bearer = HTTPBearer(auto_error=False)

def _require_admin_password(x_admin_password: Optional[str] = Header(None)) -> None:
    """FastAPI dependency — raises 401 unless X-Admin-Password header matches ADMIN_PASSWORD."""
    if x_admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid admin password")

def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """FastAPI dependency — resolves JWT to user dict or raises 401."""
    token = creds.credentials if creds else None
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = _auth.decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id, username, role, is_active FROM users WHERE id=?",
            (int(payload["sub"]),),
        ).fetchone()
    if not row or not row[3]:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return {"id": row[0], "username": row[1], "role": row[2]}

def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """FastAPI dependency — raises 403 unless user is admin."""
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

# ─── Schemas ──────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=80)
    password: str = Field(..., min_length=1)

class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=80)
    password: str = Field(..., min_length=6)
    role:     str = Field("user")

class UpdateUserRequest(BaseModel):
    password:  Optional[str] = None
    role:      Optional[str] = None
    is_active: Optional[bool] = None

class GenerateRequest(BaseModel):
    prompt:         str           = Field(..., min_length=1, max_length=2000)
    lyrics:         Optional[str] = Field(None, max_length=8000)
    duration:       float         = Field(60.0, ge=-1, le=480)  # -1 = model auto-determines from lyrics
    guidance_scale: float         = Field(7.5,  ge=1.0, le=20.0)
    infer_steps:    int           = Field(30,   ge=10,  le=100)
    seed:           int           = Field(-1)
    bpm:            Optional[int] = Field(None, ge=40, le=300)
    key:            Optional[str] = Field(None)
    language:       Optional[str] = Field("en")
    voice_id:       Optional[str] = Field(None)
    vocal_gender:   Optional[str] = Field("auto")  # auto | female | male | mixed

class QueueResponse(BaseModel):
    job_id:    str
    position:  int
    estimated_wait_minutes: float

class PresetSave(BaseModel):
    """Payload for saving a generation preset."""
    name:          str   = Field(..., min_length=1, max_length=80)
    style_tags:    List[str] = Field(default_factory=list)
    vocal_gender:  str   = Field("auto")
    duration:      float = Field(60.0)
    auto_duration: bool  = Field(False)
    guidance_scale: float = Field(7.5)
    infer_steps:   int   = Field(30)
    bpm:           Optional[int] = Field(None)
    key:           Optional[str] = Field(None)
    lang:          Optional[str] = Field("en")

# ─── Helpers ──────────────────────────────────────────────────
_VOCAL_GENDER_TAGS = {
    "female": "female vocals",
    "male":   "male vocals",
    "mixed":  "male and female vocals",
}
_VOCAL_STRIP_RE = re.compile(
    r'\b(male and female|female|male)\s+vocals?\b[,\s]*',
    re.IGNORECASE,
)

def _apply_vocal_gender(prompt: str, gender: Optional[str]) -> str:
    """Prepend the chosen vocal gender tag, stripping any existing gender term first."""
    tag = _VOCAL_GENDER_TAGS.get(gender or "auto")
    if not tag:
        return prompt
    cleaned = _VOCAL_STRIP_RE.sub("", prompt).strip().strip(",").strip()
    return f"{tag}, {cleaned}" if cleaned else tag

def get_usage_key(ip: str) -> str:
    return f"{ip}::{date.today().isoformat()}"

def get_remaining(ip: str) -> int:
    return max(0, MAX_PER_DAY - daily_usage[get_usage_key(ip)])

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host

def queue_state() -> dict:
    return {
        "queue_length": len(generation_queue),
        "active_job": {
            "job_id":   active_job["job_id"]  if active_job else None,
            "prompt":   active_job["prompt"]  if active_job else None,
            "elapsed":  int((datetime.now() - active_job["started_at"]).total_seconds()) if active_job else 0,
            "progress": active_job.get("_progress", 0.0) if active_job else 0.0,
        },
        "total_generated": total_generated,
        "total_failed":    total_failed,
    }

def _sanitize_json(obj):
    """Replace non-finite floats (Infinity, NaN) with None for valid JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj

async def broadcast(data: dict):
    payload = json.dumps(_sanitize_json(data))
    dead = []
    for ws in list(ws_clients):
        try:
            await asyncio.wait_for(ws.send_text(payload), timeout=5.0)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in ws_clients:
            ws_clients.remove(ws)
            log.debug(f"Removed dead WebSocket client ({len(ws_clients)} remaining)")

def _record_finished(job_id: str, status: str, audio_url: Optional[str] = None, error: Optional[str] = None) -> None:
    """Store a finished job so /job/{id} can answer HTTP polls after WS broadcast is gone."""
    recently_finished[job_id] = {"status": status, "audio_url": audio_url, "error": error}
    if len(recently_finished) > 100:
        del recently_finished[next(iter(recently_finished))]

def find_audio_file(task_id: str) -> Optional[Path]:
    search_dirs = [
        AUDIO_STORE_DIR,
        Path(ACESTEP_AUDIO_DIR),
        Path(__file__).parent.parent / "ACE-Step-1.5" / ".cache" / "acestep" / "tmp" / "api_audio",
    ]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for ext in ("mp3", "wav", "flac", "ogg"):
            candidate = search_dir / f"{task_id}.{ext}"
            if candidate.exists() and candidate.stat().st_size > 1000:
                return candidate
        try:
            for f in search_dir.iterdir():
                if task_id in f.name and f.suffix in (".mp3", ".wav", ".flac"):
                    if f.stat().st_size > 1000:
                        return f
        except Exception:
            pass
    return None

def find_newest_audio_after(since_timestamp: float) -> Optional[Path]:
    """Fallback: find newest audio file created after the given timestamp via filesystem scan."""
    search_dirs = [
        Path(ACESTEP_AUDIO_DIR),
        Path(__file__).parent.parent / "ACE-Step-1.5" / ".cache" / "acestep" / "tmp" / "api_audio",
    ]
    best_file  = None
    best_mtime = since_timestamp

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        try:
            for f in search_dir.iterdir():
                if f.suffix.lower() not in (".mp3", ".wav", ".flac"):
                    continue
                try:
                    stat  = f.stat()
                    mtime = stat.st_mtime
                    size  = stat.st_size
                except OSError:
                    continue
                if mtime > best_mtime and size > 10_000:
                    best_mtime = mtime
                    best_file  = f
        except Exception as e:
            log.warning(f"Audio dir scan error: {e}")

    if best_file:
        log.info(f"Newest audio candidate (fs scan): {best_file.name}")
    return best_file


class _APIConnectionError(Exception):
    """Raised when ACE-Step API cannot be reached (network error, refused connection)."""

async def _query_acestep_result(task_id: str) -> tuple:
    """Poll ACE-Step /query_result. Returns (audio_path_or_None, progress_dict).

    progress_dict keys: progress (0.0-1.0), stage (str), progress_text (str).
    Raises RuntimeError on generation failure, _APIConnectionError on network error.
    """
    prog: dict = {"progress": 0.0, "stage": "", "progress_text": ""}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"{ACESTEP_API}/query_result",
                json={"task_id_list": [task_id]},
            )
        if r.status_code != 200:
            return None, prog
        data = r.json().get("data", [])
        if not data:
            return None, prog
        item = data[0]
        status = item.get("status", 0)
        prog["progress_text"] = item.get("progress_text", "") or ""
        result_raw = item.get("result", "[]")
        result_list = json.loads(result_raw) if isinstance(result_raw, str) else result_raw
        if isinstance(result_list, list) and result_list:
            first = result_list[0]
            prog["progress"] = float(first.get("progress", 0.0))
            prog["stage"]    = str(first.get("stage", "") or "")
        if status == 0:
            return None, prog  # still running
        if status == 2:
            raise RuntimeError("ACE-Step reported generation failure")
        # status == 1 → succeeded; parse audio path from result
        if isinstance(result_list, list) and result_list:
            file_path = result_list[0].get("file", "")
            if file_path:
                p = Path(file_path)
                if p.exists() and p.stat().st_size > 10_000:
                    prog["progress"] = 1.0
                    return p, prog
    except (RuntimeError, _APIConnectionError):
        raise
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
        raise _APIConnectionError(str(e))
    except Exception as e:
        log.debug(f"query_result poll error: {e}")
    return None, prog

# ─── Background queue worker ──────────────────────────────────
class _JobCancelled(Exception):
    """Raised inside queue_worker when the user cancels the active job."""


async def queue_worker():
    global active_job, total_generated, total_failed
    log.info("Queue worker started")
    log.info(f"Audio dir: {ACESTEP_AUDIO_DIR}")

    while True:
        await asyncio.sleep(1)
        if active_job is not None or not generation_queue:
            continue

        try:
            job = generation_queue.pop(0)
        except IndexError:
            continue  # queue was cleared between the check and the pop
        _persist_queue()
        active_job = job
        active_job["started_at"] = datetime.now()
        active_job["status"]     = "generating"

        try:
            await broadcast({"type": "queue_update", **queue_state()})
            await broadcast({"type": "job_started", "job_id": job["job_id"]})
            log.info(f"Processing job {job['job_id']} — {job['prompt'][:60]}...")
            # Scale timeout with requested duration: 15 s of wall-clock per second of audio,
            # minimum 600 s, hard cap at 3600 s (1 h) for very long pieces.
            requested_dur = job.get("duration", -1) or -1
            job_poll_timeout = (
                max(POLL_TIMEOUT, int(requested_dur * 15))
                if requested_dur > 0
                else POLL_TIMEOUT
            )
            job_poll_timeout = min(job_poll_timeout, 3600)

            task_id = str(uuid.uuid4())
            payload = {
                "task_id":         task_id,
                "task_type":       job.get("task_type", "text2music"),
                "caption":         job["prompt"],
                "lyrics":          strip_section_headers(job.get("lyrics") or ""),
                "duration":        job["duration"],
                "guidance_scale":  job["guidance_scale"],
                "inference_steps": job["infer_steps"],
                "seed":            job["seed"],
                "batch_size":      1,
                "use_erg_tag":     True,
                "use_cot_metas":   True,
                "use_cot_caption": False,  # Preserve user's caption verbatim; LLM rewrites drop gender terms
                "lm_mode":         "think",
            }
            if job.get("bpm"):       payload["bpm"]       = job["bpm"]
            if job.get("key"):       payload["key"]        = job["key"]
            if job.get("language"):  payload["language"]   = job["language"]
            if job.get("src_audio"): payload["src_audio"]  = job["src_audio"]

            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(f"{ACESTEP_API}/release_task", json=payload)
                r.raise_for_status()
                submit_data    = r.json()
                log.info(f"Submit response: {json.dumps(submit_data)[:300]}")
                server_task_id = task_id
                if isinstance(submit_data.get("data"), dict):
                    server_task_id = submit_data["data"].get("task_id", task_id)
                elif submit_data.get("task_id"):
                    server_task_id = submit_data["task_id"]

            log.info(f"Task submitted — server_task_id={server_task_id}")

            submit_time = datetime.now().timestamp()
            poll_start  = asyncio.get_running_loop().time()
            audio_url   = None
            seed_out    = job["seed"]
            consecutive_api_failures = 0

            await asyncio.sleep(6)  # give ACE-Step time to start the job
            if job["job_id"] in _cancelled_jobs:
                _cancelled_jobs.discard(job["job_id"])
                raise _JobCancelled()

            while True:
                if job["job_id"] in _cancelled_jobs:
                    _cancelled_jobs.discard(job["job_id"])
                    raise _JobCancelled()

                # Use real wall-clock elapsed so httpx timeouts don't inflate the counter.
                elapsed_secs = asyncio.get_running_loop().time() - poll_start + 6
                if elapsed_secs >= job_poll_timeout:
                    break

                await asyncio.sleep(POLL_INTERVAL)
                elapsed_secs = asyncio.get_running_loop().time() - poll_start + 6

                # Primary: ask ACE-Step API for result + real progress
                try:
                    audio_file, prog = await _query_acestep_result(server_task_id)
                    consecutive_api_failures = 0  # successful API contact
                    active_job["_progress"] = prog.get("progress", 0.0)
                except RuntimeError as e:
                    raise Exception(str(e))
                except _APIConnectionError:
                    consecutive_api_failures += 1
                    audio_file, prog = None, {}

                await broadcast({
                    "type":          "job_progress",
                    "job_id":        job["job_id"],
                    "elapsed":       int(elapsed_secs),
                    "status":        "generating",
                    "progress":      prog.get("progress", 0.0),
                    "stage":         prog.get("stage", ""),
                    "progress_text": prog.get("progress_text", ""),
                })

                # Fallback: filesystem scan (handles edge-cases where task_id mapping differs)
                if not audio_file:
                    audio_file = find_newest_audio_after(submit_time)

                # Stability gate — applies to BOTH primary API path and filesystem fallback.
                # A 3-min WAV is ~30MB; it exceeds any simple size threshold long before it's
                # fully written.  We only accept the file once its size has stopped changing
                # across two consecutive poll cycles (i.e. at least POLL_INTERVAL seconds at
                # the same size), which confirms ACE-Step has finished flushing to disk.
                if audio_file:
                    try:
                        current_sz = audio_file.stat().st_size
                    except OSError:
                        current_sz = 0
                    _key = str(audio_file)
                    prev_sz = _audio_scan_sizes.get(_key)
                    _audio_scan_sizes[_key] = current_sz
                    if current_sz < 50_000 or prev_sz is None or prev_sz != current_sz:
                        log.info(
                            f"Audio found but still writing "
                            f"(prev={prev_sz} cur={current_sz}) — waiting…"
                        )
                        audio_file = None  # not stable yet, continue polling

                if audio_file:
                    _cache_set(audio_file_map, server_task_id, audio_file)
                    _cache_set(audio_file_map, job["job_id"],  audio_file)
                    audio_url = f"/audio/{job['job_id']}"
                    log.info(f"Audio ready after {elapsed_secs:.0f}s — {audio_file}")
                    break

                # After 5 consecutive API call failures (~15s), check if ACE-Step is still up
                if consecutive_api_failures >= 5:
                    try:
                        async with httpx.AsyncClient(timeout=4) as client:
                            hc = await client.get(f"{ACESTEP_API}/health")
                        if hc.status_code != 200:
                            raise Exception("ACE-Step returned unhealthy status")
                        consecutive_api_failures = 0  # it's up, just slow
                    except Exception as hc_err:
                        raise Exception(f"ACE-Step backend unreachable after {elapsed_secs:.0f}s: {hc_err}")

                log.info(f"Waiting for audio... elapsed={elapsed_secs:.0f}s")

            if not audio_url:
                raise Exception(f"Generation timed out after {job_poll_timeout}s")

            # ── Voice pipeline post-processing ──────────────────
            if job.get("voice_id"):
                voice_id  = job["voice_id"]
                profile   = voice_profile_registry.get(voice_id, {})
                vtype     = profile.get("type", "svc") if isinstance(profile, dict) else "svc"
                voice_ref = (
                    RVC_MODELS_DIR / f"{voice_id}.pth"
                    if vtype == "rvc"
                    else VOICE_PROFILES_DIR / f"{voice_id}.wav"
                )
                if voice_ref.exists() and audio_file:
                    final_out = VOICE_SONGS_DIR / f"{job['job_id']}.wav"
                    log.info(f"[voice] Running {vtype.upper()} pipeline for job {job['job_id']}…")
                    await broadcast({
                        "type":    "job_progress",
                        "job_id":  job["job_id"],
                        "elapsed": int(elapsed_secs),
                        "status":  "applying voice",
                    })
                    try:
                        env = {**os.environ, "PYTHONUTF8": "1", "VOICE_TYPE": vtype}
                        if vtype == "rvc":
                            env["RVC_MODEL_PATH"] = str(voice_ref)
                            idx = RVC_MODELS_DIR / f"{voice_id}.index"
                            if idx.exists():
                                env["RVC_INDEX_PATH"] = str(idx)
                        vproc = await asyncio.create_subprocess_exec(
                            sys.executable, str(VOICE_PIPELINE),
                            str(audio_file), str(voice_ref), str(final_out),
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            env=env,
                        )
                        process_guard.register(vproc)
                        try:
                            _, vstderr = await asyncio.wait_for(
                                vproc.communicate(), timeout=VOICE_PIPELINE_TIMEOUT
                            )
                        finally:
                            process_guard.unregister(vproc)
                        if vproc.returncode == 0 and final_out.exists():
                            _cache_set(audio_file_map, server_task_id, final_out)
                            _cache_set(audio_file_map, job["job_id"],  final_out)
                            audio_url = f"/audio/{job['job_id']}"
                            log.info(f"[voice] {vtype.upper()} done → {final_out.name}")
                        else:
                            err = vstderr.decode(errors="replace")[-400:]
                            log.error(f"[voice] {vtype.upper()} pipeline failed:\n{err}")
                            # Fall through — serve original generation without voice
                    except asyncio.TimeoutError:
                        log.error(f"[voice] {vtype.upper()} pipeline timed out — serving original")
                else:
                    log.warning(f"[voice] Profile {voice_id} ({vtype}) not found, skipping")

            job["status"]    = "completed"
            job["audio_url"] = audio_url
            job["seed_out"]  = seed_out
            job["task_id"]   = server_task_id
            total_generated += 1

            # Save to database
            save_track_to_db(job, str(audio_file_map.get(server_task_id, "")))
            _record_finished(job["job_id"], "completed", audio_url=audio_url)

            await broadcast({
                "type":      "job_completed",
                "job_id":    job["job_id"],
                "audio_url": audio_url,
                "seed":      seed_out,
            })
            log.info(f"Job {job['job_id']} completed")

        except _JobCancelled:
            log.info(f"Job {job['job_id']} cancelled by user")
            job["status"] = "cancelled"
            _record_finished(job["job_id"], "cancelled")
            await broadcast({"type": "job_cancelled", "job_id": job["job_id"]})
        except Exception as e:
            log.error(f"Job {job['job_id']} failed: {e}")
            job["status"] = "failed"
            job["error"]  = str(e)
            total_failed += 1
            _record_finished(job["job_id"], "failed", error=str(e))
            await broadcast({
                "type":    "job_failed",
                "job_id":  job["job_id"],
                "error":   str(e),
            })
        finally:
            active_job = None
            await broadcast({"type": "queue_update", **queue_state()})

async def _watched_queue_worker():
    """Run queue_worker and restart it automatically if it crashes."""
    global active_job
    while True:
        try:
            await queue_worker()
        except Exception as e:
            log.error(f"[worker] Crashed unexpectedly: {e} — restarting in 5s")
            active_job = None  # clear any stuck state left by the crash
            await asyncio.sleep(5)

@app.on_event("startup")
async def startup():
    init_db()
    _ensure_admin_user()
    _load_queue_from_db()
    _load_voice_index()
    # Reap any worker processes left behind by a previous crash. They would
    # otherwise hold the GPU / DB locks and silently block new generations.
    orphans = process_guard.find_orphans()
    if orphans:
        names = ", ".join(f"{o['name']}({o['pid']})" for o in orphans[:5])
        log.warning(f"Found {len(orphans)} orphan worker(s) from a previous run: {names}")
        process_guard.kill_orphans()
    asyncio.create_task(_watched_queue_worker())
    asyncio.create_task(_auto_pull_ollama_model())
    log.info(f"MAZ started — audio dir: {ACESTEP_AUDIO_DIR}")
    try:
        import socket as _sock
        _s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
        _s.connect(("8.8.8.8", 80))
        _lan = _s.getsockname()[0]
        _s.close()
        log.info(f"Mobile access (same Wi-Fi): http://{_lan}:3000")
    except Exception:
        pass


@app.on_event("shutdown")
async def shutdown():
    """Kill every tracked child (and their descendants) so nothing leaks."""
    killed = process_guard.kill_all()
    if killed:
        log.info(f"MAZ shutdown — terminated {killed} child process(es)")

# ─── Routes ───────────────────────────────────────────────────
@app.get("/")
async def root():
    import time
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    # Inject cache-busting version so browser always loads the latest JS/CSS
    v = int(time.time())
    html = html.replace('src="/static/app.js"',    f'src="/static/app.js?v={v}"')
    html = html.replace('href="/static/style.css"', f'href="/static/style.css?v={v}"')
    return HTMLResponse(html)

@app.get("/admin")
async def admin_page(_: None = Depends(_require_admin_password)):
    html = (STATIC_DIR / "admin.html").read_text()
    return HTMLResponse(html)

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{ACESTEP_API}/health")
            backend_ok = r.status_code == 200
    except Exception:
        backend_ok = False
    audio_dir = Path(ACESTEP_AUDIO_DIR)
    return {
        "status":           "ok",
        "acestep_backend":  "ok" if backend_ok else "unreachable",
        "audio_dir":        str(audio_dir),
        "audio_dir_exists": audio_dir.exists(),
        "queue_length":     len(generation_queue),
        "active":           active_job is not None,
    }

# ─── Auth endpoints ───────────────────────────────────────────
@app.post("/auth/login")
async def auth_login(req: LoginRequest):
    """Return a JWT token for valid credentials."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id, password_hash, role, is_active FROM users WHERE username=?",
            (req.username,),
        ).fetchone()
    if not row or not row[3]:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not _auth.verify_password(req.password, row[1]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = _auth.create_token(row[0], row[2])
    return {"token": token, "username": req.username, "role": row[2], "user_id": row[0]}

@app.get("/auth/me")
async def auth_me(user: dict = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return user

@app.get("/auth/users")
async def auth_list_users(admin: dict = Depends(require_admin)):
    """List all users (admin only)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, username, role, is_active, created_at FROM users ORDER BY id"
        ).fetchall()
        # Attach per-user track counts
        counts = {
            r[0]: r[1]
            for r in conn.execute("SELECT user_id, COUNT(*) FROM tracks GROUP BY user_id").fetchall()
        }
    users = [dict(r) for r in rows]
    for u in users:
        u["track_count"] = counts.get(u["id"], 0)
    return users

@app.post("/auth/users", status_code=201)
async def auth_create_user(req: CreateUserRequest, admin: dict = Depends(require_admin)):
    """Create a new user account (admin only)."""
    if req.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="role must be 'admin' or 'user'")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
                (req.username, _auth.hash_password(req.password), req.role),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Username already taken")
    return {"created": req.username}

@app.patch("/auth/users/{user_id}")
async def auth_update_user(
    user_id: int, req: UpdateUserRequest, admin: dict = Depends(require_admin)
):
    """Update password, role, or active status for a user (admin only)."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT id FROM users WHERE id=?", (user_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        if req.password is not None:
            conn.execute(
                "UPDATE users SET password_hash=? WHERE id=?",
                (_auth.hash_password(req.password), user_id),
            )
        if req.role is not None:
            conn.execute("UPDATE users SET role=? WHERE id=?", (req.role, user_id))
        if req.is_active is not None:
            conn.execute(
                "UPDATE users SET is_active=? WHERE id=?", (int(req.is_active), user_id)
            )
        conn.commit()
    return {"updated": user_id}

@app.delete("/auth/users/{user_id}")
async def auth_delete_user(user_id: int, admin: dict = Depends(require_admin)):
    """Delete a user account (admin only). Cannot delete own account."""
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
    return {"deleted": user_id}

@app.post("/auth/change-password")
async def auth_change_password(
    req: UpdateUserRequest, user: dict = Depends(get_current_user)
):
    """Allow any logged-in user to change their own password."""
    if not req.password or len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE users SET password_hash=? WHERE id=?",
            (_auth.hash_password(req.password), user["id"]),
        )
        conn.commit()
    return {"changed": True}


@app.get("/ai/status")
async def ai_status():
    """Check if Ollama is running and which model will be used for AI features."""
    gemini_info = {"gemini_available": bool(GEMINI_API_KEY), "gemini_model": GEMINI_MODEL if GEMINI_API_KEY else None}
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            r = await client.get("http://localhost:11434/api/tags")
        if r.status_code != 200:
            return {"available": False, "reason": "Ollama returned unexpected status", "model": None, "models": [], **gemini_info}
        installed = [m["name"] for m in r.json().get("models", [])]
        model = await _pick_local_model()
        if model:
            return {"available": True, "model": model, "models": installed, "reason": None, **gemini_info}
        else:
            return {
                "available": bool(GEMINI_API_KEY),
                "model": None,
                "models": [],
                "reason": "No local models installed. Run: ollama pull qwen3:8b",
                **gemini_info,
            }
    except Exception as e:
        return {
            "available": bool(GEMINI_API_KEY),
            "model": None,
            "models": [],
            "reason": f"Ollama not running. Install from https://ollama.ai — {e}",
            **gemini_info,
        }

@app.get("/usage")
async def usage(request: Request):
    ip   = get_client_ip(request)
    used = daily_usage[get_usage_key(ip)]
    return {"remaining": get_remaining(ip), "limit": MAX_PER_DAY, "used": used}

@app.post("/generate", response_model=QueueResponse)
async def generate(
    req: GenerateRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    ip = get_client_ip(request)
    if get_remaining(ip) <= 0:
        raise HTTPException(status_code=429, detail=f"Daily limit of {MAX_PER_DAY} generations reached. Try again tomorrow.")
    daily_usage[get_usage_key(ip)] += 1

    job = {
        "job_id":         str(uuid.uuid4()),
        "ip":             ip,
        "user_id":        user["id"],
        "prompt":         _apply_vocal_gender(req.prompt, req.vocal_gender),
        "lyrics":         req.lyrics,
        "duration":       req.duration,
        "guidance_scale": req.guidance_scale,
        "infer_steps":    req.infer_steps,
        "seed":           req.seed,
        "bpm":            req.bpm,
        "key":            req.key,
        "language":       req.language or "en",
        "voice_id":       req.voice_id or None,
        "status":         "queued",
        "queued_at":      datetime.now(),
        "started_at":     None,
        "audio_url":      None,
        "error":          None,
    }

    generation_queue.append(job)
    _persist_queue()
    position = len(generation_queue)

    active_elapsed   = int((datetime.now() - active_job["started_at"]).total_seconds()) if active_job else 0
    active_remaining = max(0, (active_job["duration"] * 0.5) - active_elapsed) if active_job else 0
    # Sum durations of all jobs ahead of this one in the queue (exclude the job we just appended)
    queued_ahead_seconds = sum(
        (j.get("duration") or 60.0) * 0.5
        for j in generation_queue[:-1]
    )
    wait_seconds  = active_remaining + queued_ahead_seconds
    wait_minutes  = round(wait_seconds / 60, 1)

    await broadcast({"type": "queue_update", **queue_state()})
    log.info(f"Job {job['job_id']} queued position={position} ip={ip}")

    return QueueResponse(
        job_id=job["job_id"],
        position=position,
        estimated_wait_minutes=wait_minutes,
    )

@app.get("/job/{job_id}")
async def get_job(job_id: str):
    for job in generation_queue:
        if job["job_id"] == job_id:
            return {"status": "queued", "position": generation_queue.index(job) + 1}
    if active_job and active_job["job_id"] == job_id:
        elapsed = int((datetime.now() - active_job["started_at"]).total_seconds())
        return {"status": "generating", "elapsed": elapsed}
    if job_id in recently_finished:
        return recently_finished[job_id]
    raise HTTPException(status_code=404, detail="Job not found")

@app.delete("/job/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a queued or active generation job."""
    global generation_queue
    before = len(generation_queue)
    generation_queue = [j for j in generation_queue if j["job_id"] != job_id]
    if len(generation_queue) < before:
        _persist_queue()
        await broadcast({"type": "queue_update", **queue_state()})
        log.info(f"Job {job_id} removed from queue by user")
        return {"cancelled": job_id}
    if active_job and active_job["job_id"] == job_id:
        _cancelled_jobs.add(job_id)
        log.info(f"Job {job_id} marked for cancellation")
        return {"cancelled": job_id}
    raise HTTPException(status_code=404, detail="Job not found")

@app.get("/audio/{task_id}")
async def get_audio(task_id: str, request: Request):
    """Serve audio with proper range request support for full playback."""
    audio_file = audio_file_map.get(task_id)
    if not audio_file:
        audio_file = find_audio_file(task_id)
        if audio_file:
            _cache_set(audio_file_map, task_id, audio_file)
    if not audio_file:
        # Library playback: task_id is the job_id — look up the stored audio path in the DB.
        # (audio_file_map is keyed by server_task_id which differs from job_id, so the
        # map lookup above always misses for library tracks after a server restart.)
        try:
            with sqlite3.connect(DB_PATH) as conn:
                # Direct job_id lookup
                row = conn.execute(
                    "SELECT audio_path, voice_sections_path FROM tracks WHERE job_id=?", (task_id,)
                ).fetchone()
                if not row and task_id.startswith("sections_"):
                    # sections_{job_id} key — look up via voice_sections_path
                    real_job_id = task_id[len("sections_"):]
                    row = conn.execute(
                        "SELECT audio_path, voice_sections_path FROM tracks WHERE job_id=?",
                        (real_job_id,)
                    ).fetchone()
                    if row:
                        # Prefer sections path for this key
                        row = (row[1], row[1])
            if row:
                col = row[1] if task_id.startswith("sections_") and row[1] else row[0]
                if col:
                    p = Path(col)
                    if p.exists() and p.stat().st_size > 1000:
                        audio_file = p
                        _cache_set(audio_file_map, task_id, audio_file)
        except Exception as e:
            log.debug(f"DB audio lookup error: {e}")
    if not audio_file or not Path(audio_file).exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    path       = Path(audio_file)
    ext        = path.suffix.lstrip(".")
    
    file_size  = path.stat().st_size
    media_type = "audio/mpeg" if ext == "mp3" else f"audio/{ext}"

    range_header = request.headers.get("range", "").strip()
    if range_header and range_header.startswith("bytes="):
        try:
            parts = range_header[6:].split("-")
            start = int(parts[0]) if parts[0] else 0
            end   = int(parts[1]) if parts[1] else file_size - 1
        except Exception:
            start, end = 0, file_size - 1
        start  = max(0, min(start, file_size - 1))
        end    = max(start, min(end, file_size - 1))
        length = end - start + 1

        
        async def ranged():
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            ranged(),
            status_code=206,
            media_type=media_type,
            headers={
                "Content-Range":       f"bytes {start}-{end}/{file_size}",
                "Content-Length":      str(length),
                "Accept-Ranges":       "bytes",
                "Cache-Control":       "no-cache",
                "Content-Disposition": f"inline; filename=maz_{task_id[:8]}.{ext}",
            },
        )

    async def full():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        full(),
        status_code=200,
        media_type=media_type,
        headers={
            "Content-Length":      str(file_size),
            "Accept-Ranges":       "bytes",
            "Cache-Control":       "no-cache",
            "Content-Disposition": f"inline; filename=maz_{task_id[:8]}.{ext}",
        },
    )

# ─── Format conversion + Mastering ───────────────────────────
def _resolve_audio(task_id: str) -> Optional[Path]:
    """Return the audio Path for task_id using the same strategy as /audio/{task_id}.

    Checks audio_file_map, filesystem scan, then DB (both audio_path and
    voice_sections_path columns). Caches successful lookups back into
    audio_file_map so repeated calls are instant.
    """
    audio_file = audio_file_map.get(task_id)
    if audio_file and Path(audio_file).exists():
        return Path(audio_file)

    found = find_audio_file(task_id)
    if found:
        _cache_set(audio_file_map, task_id, found)
        return Path(found)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT audio_path, voice_sections_path FROM tracks WHERE job_id=?",
                (task_id,)
            ).fetchone()
        if row:
            for col in (row[1], row[0]):   # prefer voice_sections_path
                if col:
                    p = Path(col)
                    if p.exists() and p.stat().st_size > 1000:
                        _cache_set(audio_file_map, task_id, p)
                        return p
    except Exception:
        pass
    return None


def _convert_audio(src: Path, fmt: str) -> bytes:
    """Convert src WAV to fmt ('mp3' or 'flac') and return raw bytes.

    MP3 conversion uses ffmpeg (always present via audio-separator dependency).
    FLAC conversion uses soundfile (no extra dependency).
    """
    import subprocess, shutil

    fmt = fmt.lower()
    if fmt not in ("mp3", "flac"):
        raise ValueError(f"Unsupported format: {fmt}")

    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tf:
        out_path = Path(tf.name)

    try:
        if fmt == "flac":
            import numpy as np
            import soundfile as sf
            data, sr = sf.read(str(src))
            sf.write(str(out_path), data, sr, format="FLAC", subtype="PCM_24")
        else:  # mp3
            ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
            subprocess.run(
                [ffmpeg, "-y", "-i", str(src), "-codec:a", "libmp3lame",
                 "-qscale:a", "2", str(out_path)],
                capture_output=True, check=True, timeout=120,
            )
        return out_path.read_bytes()
    finally:
        out_path.unlink(missing_ok=True)


@app.get("/download/{task_id}")
async def download_track_fmt(task_id: str, format: str = "wav"):
    """Download a track in wav, mp3, or flac format."""
    fmt = format.lower()
    if fmt not in ("wav", "mp3", "flac"):
        raise HTTPException(status_code=400, detail="format must be wav, mp3, or flac")

    src = _resolve_audio(task_id)
    if not src:
        raise HTTPException(status_code=404, detail="Audio file not found")

    if fmt == "wav":
        data = src.read_bytes()
        media_type = "audio/wav"
    else:
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                data = await loop.run_in_executor(pool, _convert_audio, src, fmt)
        except Exception as e:
            log.error(f"Format conversion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        media_type = "audio/mpeg" if fmt == "mp3" else "audio/flac"

    filename = f"maz_{task_id[:8]}.{fmt}"
    return StreamingResponse(
        io.BytesIO(data),
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Length": str(len(data)),
        },
    )


def _master_audio(src: Path) -> bytes:
    """Normalise loudness to -14 LUFS (ITU-R BS.1770-4) and apply true-peak limiting.

    Uses pyloudnorm for accurate integrated loudness measurement.
    Returns WAV bytes of the mastered file.
    """
    import io as _io
    import numpy as np
    import soundfile as sf
    import pyloudnorm as pyln

    data, sr = sf.read(str(src), always_2d=True)   # (samples, channels)

    TARGET_LUFS = -14.0
    meter = pyln.Meter(sr)  # ITU-R BS.1770-4 meter
    current_lufs = meter.integrated_loudness(data)

    # integrated_loudness returns -inf for silence; guard against that
    if not math.isfinite(current_lufs):
        current_lufs = -70.0

    gain_db = TARGET_LUFS - current_lufs
    # Clamp: never boost more than +12 dB or cut more than -24 dB
    gain_db = max(-24.0, min(12.0, gain_db))
    gain_linear = 10 ** (gain_db / 20.0)
    data = data * gain_linear

    # True-peak limiter at -1 dBFS
    TP_CEIL = 10 ** (-1.0 / 20.0)
    peak = float(np.max(np.abs(data)))
    if peak > TP_CEIL:
        data = data * (TP_CEIL / peak)

    buf = _io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


@app.get("/master/{task_id}")
async def master_track(task_id: str):
    """Return a loudness-normalised (-14 LUFS) + true-peak-limited WAV."""
    src = _resolve_audio(task_id)
    if not src:
        raise HTTPException(status_code=404, detail="Audio file not found")
    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            data = await loop.run_in_executor(pool, _master_audio, src)
    except Exception as e:
        log.error(f"Mastering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    filename = f"maz_{task_id[:8]}_mastered.wav"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Length": str(len(data)),
        },
    )


# ─── WebSocket ─────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.append(websocket)
    await websocket.send_json({"type": "queue_update", **queue_state()})
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in ws_clients:
            ws_clients.remove(websocket)

# ─── Admin API ────────────────────────────────────────────────
@app.get("/admin/stats")
async def admin_stats(_: None = Depends(_require_admin_password)):
    return {
        "queue": [
            {
                "job_id":    j["job_id"],
                "ip":        j["ip"],
                "prompt":    j["prompt"][:60],
                "duration":  j["duration"],
                "queued_at": j["queued_at"].isoformat(),
            } for j in generation_queue
        ],
        "active_job": {
            "job_id":   active_job["job_id"]      if active_job else None,
            "ip":       active_job["ip"]           if active_job else None,
            "prompt":   active_job["prompt"][:60]  if active_job else None,
            "duration": active_job["duration"]     if active_job else None,
            "elapsed":  int((datetime.now() - active_job["started_at"]).total_seconds()) if active_job else 0,
        },
        "total_generated": total_generated,
        "total_failed":    total_failed,
        "daily_usage":     dict(daily_usage),
        "max_per_day":     MAX_PER_DAY,
        "connected_users": len(ws_clients),
        "audio_dir":       ACESTEP_AUDIO_DIR,
    }

@app.delete("/admin/active")
async def admin_clear_active(_: None = Depends(_require_admin_password)):
    """Force-clear a stuck active job so the queue can proceed."""
    global active_job
    if active_job is None:
        return {"cleared": None, "message": "No active job"}
    cleared_id = active_job["job_id"]
    active_job = None
    await broadcast({"type": "queue_update", **queue_state()})
    log.warning(f"Admin force-cleared active job {cleared_id}")
    return {"cleared": cleared_id}

@app.delete("/admin/queue/{job_id}")
async def admin_cancel(job_id: str, _: None = Depends(_require_admin_password)):
    global generation_queue
    before = len(generation_queue)
    generation_queue = [j for j in generation_queue if j["job_id"] != job_id]
    if len(generation_queue) < before:
        _persist_queue()
        await broadcast({"type": "queue_update", **queue_state()})
        return {"cancelled": job_id}
    raise HTTPException(status_code=404, detail="Job not in queue")

@app.delete("/admin/queue")
async def admin_clear_queue(_: None = Depends(_require_admin_password)):
    global generation_queue
    generation_queue = []
    _persist_queue()
    await broadcast({"type": "queue_update", **queue_state()})
    return {"cleared": True}


@app.post("/admin/panic")
async def admin_panic(_: None = Depends(_require_admin_password)):
    """Nuclear option: kill every tracked child + any orphan workers, drop the
    active job, and clear the queue. Use when generation is wedged and a server
    restart hasn't helped (typically: prior crash left worker processes holding
    the GPU or DB locks)."""
    global active_job, generation_queue
    cleared_active = active_job["job_id"] if active_job else None
    queued_count = len(generation_queue)
    active_job = None
    generation_queue = []
    _persist_queue()
    children = process_guard.kill_all()
    orphans = process_guard.kill_orphans()
    await broadcast({"type": "queue_update", **queue_state()})
    log.warning(
        f"Admin /panic — children={children} orphans={orphans} "
        f"active={cleared_active} queued={queued_count}"
    )
    return {
        "children_killed": children,
        "orphans_killed":  orphans,
        "active_cleared":  cleared_active,
        "queue_cleared":   queued_count,
    }

# ─── AI endpoints ─────────────────────────────────────────────
class AIRequest(BaseModel):
    prompt: str
    language: Optional[str] = None   # ISO-639-1 hint: "kk", "ru", "en", etc.
    mode: Optional[str] = "lyrics"   # "lyrics" or "enhance"

# ─── Kazakh language helpers ──────────────────────────────────
# Kazakh has 9 letters absent from Russian: ә ғ қ ң ө ү ұ і (+ uppercase variants).
# Scoring their frequency vs all Cyrillic reliably separates Kazakh from Russian output.
_KK_SPECIFIC = frozenset("әғқңөүұіӘҒҚҢӨҮҰІ")
_CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁәғқңөүұіӘҒҚҢӨҮҰІ]")

def _kazakh_score(text: str) -> float:
    """Fraction of Cyrillic characters that are Kazakh-specific (not shared with Russian).

    Score < 0.03 means fewer than 3% Kazakh-specific letters → likely Russian output.
    Returns 1.0 for very short texts (can't judge reliably).
    """
    cyrillic = _CYRILLIC_RE.findall(text)
    if len(cyrillic) < 30:
        return 1.0
    return sum(1 for c in cyrillic if c in _KK_SPECIFIC) / len(cyrillic)


_KK_SYSTEM_PROMPT = (
    "Сен — қазақ тілінде жазатын кәсіби ақын және әнші.\n"
    "Барлық мәтінді тек таза қазақ тілінде жаз. Орысша, ағылшынша немесе аралас тілде жазуға қатаң тыйым салынады.\n\n"
    "Қазақ тілінің грамматикалық ережелері:\n"
    "• Қазақша арнайы дыбыстар: ә, ғ, қ, ң, ө, ү, ұ, і — міндетті түрде дұрыс қолдан\n"
    "• Септік жалғаулары: -ның/-нің, -ға/-ге/-қа/-ке, -да/-де/-та/-те, -дан/-ден/-тан/-тен\n"
    "• Етістік жұрнақтары: -ады/-еді, -ған/-ген, -мақ/-мек, -ып/-іп\n"
    "• Ұйқас: ABAB немесе AABB — жол соңдары ұйқасуы керек; ырғақ табиғи болуы керек\n\n"
    "Қатаң тыйым:\n"
    "• Орысша сөздер (любовь, сердце, небо, душа, мечта және т.б.) — жазуға болмайды\n"
    "• Транслитерация (qazan, zhuldyz, zhürek) — жазуға болмайды\n"
    "• Орысша грамматика (-ого, -его, -ать, -ить жалғаулары) — жазуға болмайды\n\n"
    "Шығармашылық бостандық: тақырыпқа сай өз сөздеріңді таңда — бекітілген сөз тізімі жоқ.\n"
    "Тек ән мәтінін жаз — ешқандай түсіндірме, аударма немесе комментарий болмасын."
)

_KK_EXAMPLE_VERSE = (
    "Мысал (осы үлгіде жаз):\n"
    "[Verse 1]\n"
    "Далада жел соғады,\n"
    "Жүрегім сенді сағынады,\n"
    "Ай нұры жолды жарытады,\n"
    "Жан сезімім шарпып тұрады.\n\n"
)


def _build_kazakh_lyrics_prompt(user_prompt: str, include_example: bool = False) -> str:
    """Build the generation request fully in Kazakh.

    Writing the user message in Kazakh is the single biggest quality lever:
    LLMs match the language of the user turn, not just the system prompt.
    """
    example = _KK_EXAMPLE_VERSE if include_example else ""
    return (
        f"{example}"
        f"Мына тақырыпта толық ән мәтіні жаз: {user_prompt}\n\n"
        "Міндетті құрылым — әр бөлімді толық, ұйқасты жолдармен жаз:\n"
        "[Verse 1] — 6-8 жол\n"
        "[Chorus] — 4-6 жол\n"
        "[Verse 2] — 6-8 жол\n"
        "[Chorus] — 4-6 жол\n"
        "[Bridge] — 4-6 жол\n"
        "[Chorus] — 4-6 жол\n"
        "[Outro] — 2-4 жол\n\n"
        "Тек таза қазақша ән мәтіні жаз. Орысша немесе ағылшынша сөз жазба."
    )


# ─── Ollama auto-pull ─────────────────────────────────────────
# Default preference list (English-first workloads).
_LOCAL_MODEL_PREFERENCE = [
    "qwen3:8b",
    "qwen3:4b",
    "mistral",
    "qwen2.5:7b",
    "qwen2.5:3b",
    "qwen2.5:1.5b",
    "deepseek-r1:7b",
    "llama3.2:3b",
    "llama3.2:1b",
    "gemma2:2b",
    "phi3:mini",
    "deepseek-r1:1.5b",
    "smollm2:1.7b",
]

# Kazakh and Russian lyrics require a model with strong multilingual coverage.
# qwen3:8b is the best local model for Kazakh (pull with: ollama pull qwen3:8b).
# qwen3 and deepseek-r1 use <think>...</think> tags which are stripped below.
# mistral is ranked last — it produces gibberish in Kazakh.
_MULTILINGUAL_MODEL_PREFERENCE = [
    "qwen3:8b",
    "qwen3:4b",
    "qwen2.5:7b",
    "qwen2.5:3b",
    "qwen2.5:1.5b",
    "deepseek-r1:7b",
    "llama3.2:3b",
    "llama3.2:1b",
    "gemma2:2b",
    "deepseek-r1:1.5b",
    "mistral",
    "phi3:mini",
    "smollm2:1.7b",
]

async def _pick_local_models(language: str = "") -> list[str]:
    """
    Query Ollama for installed models and return them in preference order.
    For Kazakh (kk), uses a multilingual-optimised ranking (qwen3 first).
    All models are local — no cloud models are used.
    Returns an empty list if Ollama is unreachable.
    """
    # Non-Latin-script and multilingual languages use the multilingual ranking
    # (qwen first, mistral last — mistral produces gibberish in Kazakh/Russian/Turkish).
    _MULTILINGUAL_LANGS = {"kk", "kazakh", "ru", "russian", "tr", "turkish",
                           "zh", "chinese", "ja", "japanese", "ko", "korean",
                           "ar", "arabic", "he", "hebrew", "uk", "ukrainian"}
    preference = (
        _MULTILINGUAL_MODEL_PREFERENCE
        if language in _MULTILINGUAL_LANGS
        else _LOCAL_MODEL_PREFERENCE
    )
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            r = await client.get("http://localhost:11434/api/tags")
        if r.status_code != 200:
            return []
        installed_names = [m["name"] for m in r.json().get("models", [])]
        log.info(f"Ollama installed models: {installed_names} (language hint: '{language}')")
        # Build ordered list: preferred models first, then any remaining installed
        ordered: list[str] = []
        seen: set[str] = set()
        for preferred in preference:
            for installed in installed_names:
                if installed not in seen and (
                    installed == preferred
                    or installed.startswith(preferred.split(":")[0] + ":")
                ):
                    ordered.append(installed)
                    seen.add(installed)
        # Append any installed models not in the preference list
        for installed in installed_names:
            if installed not in seen:
                ordered.append(installed)
                seen.add(installed)
        return ordered
    except Exception as e:
        log.info(f"Ollama unreachable: {e}")
    return []


async def _pick_local_model(language: str = "") -> Optional[str]:
    """Convenience wrapper: return the single best model, or None."""
    models = await _pick_local_models(language)
    return models[0] if models else None


# The model to auto-pull if none is installed.
_AUTO_PULL_MODEL = "qwen3:8b"

async def _auto_pull_ollama_model():
    """
    Background task: if Ollama is running but has no models installed,
    automatically pull the smallest preferred model.
    Runs once at server startup, non-blocking.
    Requires internet only for the initial pull.
    """
    await asyncio.sleep(3)  # Wait for server to finish starting up

    # Check if Ollama is reachable
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get("http://localhost:11434/api/tags")
        if r.status_code != 200:
            log.info("[AI] Ollama not running — skipping auto-pull")
            return
        installed = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        log.info("[AI] Ollama not reachable — skipping auto-pull")
        return

    if installed:
        log.info(f"[AI] Ollama has models installed: {installed} — no pull needed")
        return

    log.info(f"[AI] No models found. Auto-pulling '{_AUTO_PULL_MODEL}' in background...")
    log.info(f"[AI] This requires internet once. ~1.1 GB download. Please wait...")

    try:
        last_logged_pct = -1
        async with httpx.AsyncClient(timeout=1800) as client:  # 30 min max
            async with client.stream(
                "POST",
                "http://localhost:11434/api/pull",
                json={"name": _AUTO_PULL_MODEL, "stream": True},
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        import json as _json
                        data = _json.loads(line)
                        status = data.get("status", "")
                        completed = data.get("completed", 0)
                        total = data.get("total", 0)
                        if total > 0:
                            pct = int((completed / total) * 100)
                            # Log every 10%
                            if pct >= last_logged_pct + 10:
                                last_logged_pct = pct
                                mb_done = completed / 1_048_576
                                mb_total = total / 1_048_576
                                log.info(f"[AI] Pulling {_AUTO_PULL_MODEL}: {pct}% ({mb_done:.0f}/{mb_total:.0f} MB)")
                        elif status and status != "pulling manifest":
                            log.info(f"[AI] Pull status: {status}")
                    except Exception:
                        pass

        log.info(f"[AI] '{_AUTO_PULL_MODEL}' pulled successfully — AI features are now ready!")

    except Exception as e:
        log.warning(f"[AI] Auto-pull failed: {e}")
        log.info(f"[AI] To pull manually, run:  ollama pull {_AUTO_PULL_MODEL}")


# ─── Gemini API (free tier) for high-quality Kazakh lyrics ────
_gemini_last_error: str = ""  # exposed to callers for richer error messages

# Tried in order on failure; all free-tier models with separate capacity pools.
_GEMINI_FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
]


async def _call_gemini(prompt: str, system_prompt: str, temperature: float = 0.7) -> Optional[str]:
    """Call Gemini REST API with automatic model fallback.

    Tries _GEMINI_FALLBACK_MODELS in order. On 503/429 ('high demand' / rate limit)
    retries once after a short delay before moving to the next model.
    Returns generated text or None if all models fail.
    Sets _gemini_last_error with the human-readable reason on failure.
    """
    global _gemini_last_error
    _gemini_last_error = ""
    if not GEMINI_API_KEY:
        _gemini_last_error = "GEMINI_API_KEY is not set"
        return None

    # Build model list: configured model first, then the rest of the fallbacks
    models_to_try = [GEMINI_MODEL] + [m for m in _GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]

    body = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 8192},
    }

    for model in models_to_try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={GEMINI_API_KEY}"
        )
        for attempt in range(2):  # 1 retry per model on transient errors
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    r = await client.post(url, json=body)

                if r.status_code == 200:
                    data = r.json()
                    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    text = "\n".join(
                        p.get("text", "") for p in parts if not p.get("thought", False)
                    ).strip()
                    if text:
                        log.info(f"Gemini response ({model}): {len(text)} chars")
                        return text
                    _gemini_last_error = f"{model}: empty response"
                    log.warning(_gemini_last_error)
                    break  # empty response — move to next model

                # 503 / 429: transient — retry once after a short wait
                if r.status_code in (429, 503) and attempt == 0:
                    try:
                        msg = r.json().get("error", {}).get("message", "")[:120]
                    except Exception:
                        msg = f"HTTP {r.status_code}"
                    log.warning(f"Gemini {model} transient error ({r.status_code}): {msg} — retrying in 3s")
                    await asyncio.sleep(3)
                    continue

                try:
                    _gemini_last_error = r.json().get("error", {}).get("message", "")[:200]
                except Exception:
                    _gemini_last_error = f"HTTP {r.status_code}"
                log.warning(f"Gemini {model} HTTP {r.status_code}: {_gemini_last_error}")
                break  # non-retryable error — move to next model

            except Exception as e:
                _gemini_last_error = f"{type(e).__name__}: {e}"
                log.warning(f"Gemini {model} exception: {_gemini_last_error}")
                break

        else:
            # Both attempts exhausted on retryable error — move to next model
            log.warning(f"Gemini {model} still failing after retry — trying next model")

    return None


def _build_lyrics_prompt(user_prompt: str) -> str:
    """Wrap any user prompt into an explicit full-song lyrics request."""
    return (
        f"Write complete, full-length song lyrics based on this idea: {user_prompt}\n\n"
        "Required structure — write every section in full, no placeholders:\n"
        "[Verse 1] — 6-8 lines\n"
        "[Chorus] — 4-6 lines (repeated feel, hook)\n"
        "[Verse 2] — 6-8 lines (new imagery, same theme)\n"
        "[Chorus] — 4-6 lines\n"
        "[Bridge] — 4-6 lines (emotional shift or contrast)\n"
        "[Chorus] — 4-6 lines\n"
        "[Outro] — 2-4 lines\n\n"
        "Output ONLY the lyrics with section labels. No explanations, no commentary."
    )


@app.post("/ai/generate")
async def ai_generate(req: AIRequest):
    """
    Routes AI requests to Gemini (for Kazakh) or local Ollama models.
    Kazakh: tries Gemini free API first (much better quality), falls back to Ollama.
    All other languages: uses local Ollama models only.
    Requires Ollama running: https://ollama.ai
    Pull a model first:  ollama pull qwen3:8b
    """
    lang = (req.language or "").lower().strip()
    is_enhance = (req.mode or "lyrics") == "enhance"

    # Enhance mode: pass the prompt straight through with no lyrics wrapping.
    if is_enhance:
        final_prompt = req.prompt
        system_prompt = "You are a helpful AI assistant. Follow the instructions in the user message exactly."
        models = await _pick_local_models(language="")
        if not models:
            if GEMINI_API_KEY:
                result = await _call_gemini(final_prompt, system_prompt)
                if result:
                    return {"result": result, "source": f"gemini/{GEMINI_MODEL}"}
            raise HTTPException(status_code=503, detail=(
                "No local AI model found. Install Ollama from https://ollama.ai, then run: ollama pull qwen3:8b"
            ))
        last_error = None
        for model in models:
            log.info(f"AI enhance: model={model}")
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    r = await client.post("http://localhost:11434/api/generate", json={
                        "model": model, "prompt": final_prompt,
                        "system": system_prompt, "stream": False, "think": False,
                        "keep_alive": 0,
                    })
                if r.status_code == 200:
                    result = re.sub(r'<think>.*?</think>', '', r.json().get("response", ""), flags=re.DOTALL).strip()
                    return {"result": result, "source": f"ollama/{model}"}
                last_error = f"Ollama returned {r.status_code}"
            except Exception as e:
                last_error = str(e)
        raise HTTPException(status_code=502, detail=f"All AI models failed. Last error: {last_error}")

    is_kazakh = lang in ("kk", "kazakh")

    # req.prompt is already the complete, language-specific prompt built by the frontend
    # (includes theme, rules, rhyme scheme, and structure in the target language).
    # Do not wrap it — double-wrapping produces contradictory instructions that break output.
    lyrics_prompt = req.prompt

    # Language-specific system prompts.
    if is_kazakh:
        system_prompt = _KK_SYSTEM_PROMPT
    elif lang in ("ru", "russian"):
        system_prompt = (
            "Ты — профессиональный русскоязычный поэт и автор текстов песен. "
            "Пиши только на русском языке с богатой образностью и чёткой рифмовкой. "
            "Обязательная структура: [Verse 1], [Chorus], [Verse 2], [Chorus], [Bridge], [Chorus], [Outro] — каждую секцию пиши полностью. "
            "Правила: используй только русский язык, соблюдай схему рифм (AABB или ABAB), "
            "пиши живые эмоциональные строки подходящие для пения. "
            "Пиши достаточно длинный текст — не менее 7 секций для 3-4 минутной песни. "
            "Никаких объяснений, переводов или комментариев — только текст песни. "
            "Выводи только текст песни, ничего больше."
        )
    elif lang in ("tr", "turkish"):
        system_prompt = (
            "Sen profesyonel bir Türkçe şair ve söz yazarısın. "
            "Yalnızca Türkçe yaz — İngilizce, Rusça veya başka bir dilde kelime kullanma. "
            "Zorunlu yapı: [Verse 1], [Chorus], [Verse 2], [Chorus], [Bridge], [Chorus], [Outro] — her bölümü tam yaz. "
            "Kurallar: "
            "1. Tüm sözler saf Türkçe olmalı — yabancı kelime veya transliterasyon yasak. "
            "2. Güçlü bir kafiye düzeni kur (AABB veya ABAB) ve her dizede doğal bir hece ritmi sağla. "
            "3. Türkçe'ye özgü ekleri kullan: -lar/-ler, -da/-de, -ın/-in, -mak/-mek, -dı/-di vb. "
            "4. Duygusal, özgün ve şarkıya uygun dizeler yaz. "
            "5. 3-4 dakikalık bir şarkı için yeterli uzunlukta yaz — en az 7 bölüm. "
            "6. Hiçbir açıklama, çeviri veya yorum ekleme — yalnızca şarkı sözleri. "
            "7. Tam ve bitmiş bir metin yaz — asla boş yer veya talimat bırakma. "
            "Yalnızca şarkı sözlerini yaz, başka hiçbir şey değil."
        )
    else:
        lang_display = lang.title() if lang else "English"
        system_prompt = (
            f"You are a professional songwriter and lyricist. "
            f"Write song lyrics ONLY in {lang_display}. "
            f"Do NOT use English or any other language — every single word must be in {lang_display}. "
            f"Use authentic vocabulary, natural rhyme schemes, and cultural style native to {lang_display}. "
            f"Always write a complete, full-length song with all sections: "
            f"[Verse 1], [Chorus], [Verse 2], [Chorus], [Bridge], [Chorus], [Outro]. "
            f"Each verse must be 6-8 lines, chorus 4-6 lines, bridge 4-6 lines. "
            f"Write enough content for a 3-4 minute song — never truncate or leave placeholders. "
            f"Never add translations, explanations, commentary, or placeholder text. "
            f"Output ONLY the song lyrics with section labels, nothing else."
        )

    # ── Kazakh: Gemini only — local models don't know Kazakh ──────
    # Never fall back to Ollama for Kazakh: small models produce meaningless
    # word-repetition and the user gets unacceptable output with no error signal.
    if is_kazakh:
        if not GEMINI_API_KEY:
            raise HTTPException(
                status_code=503,
                detail="Kazakh lyrics require the Gemini API. Set GEMINI_API_KEY in maz/.env (free key at aistudio.google.com).",
            )
        log.info(f"AI generate: Gemini ({GEMINI_MODEL}) for Kazakh")
        result = await _call_gemini(lyrics_prompt, system_prompt, temperature=0.5)
        if result:
            score = _kazakh_score(result)
            log.info(f"Gemini Kazakh score={score:.3f} ({len(result)} chars) — returning")
            return {"result": result, "source": f"gemini/{GEMINI_MODEL}"}
        # All Gemini models failed — fall back to local Ollama rather than crashing.
        # Ollama output quality for Kazakh is poor, but during a demo a degraded result
        # is far better than a hard error. Log clearly so the cause is obvious.
        log.warning(f"All Gemini models failed for Kazakh ({_gemini_last_error}) — falling back to Ollama")

    # ── Local Ollama (all languages, or Kazakh fallback) ──────
    # For Kazakh, only the best available model is tried. Models that lack
    # Kazakh coverage (llama3.2, mistral, etc.) always fail the score check,
    # so iterating through all installed models just adds N×2 reload penalties.
    all_models = await _pick_local_models(language=lang)
    models = all_models[:1] if is_kazakh else all_models

    if not models:
        # Last resort: use Gemini for any language if no local model is available
        if GEMINI_API_KEY:
            log.info(f"No local models — falling back to Gemini for language '{lang}'")
            result = await _call_gemini(lyrics_prompt, system_prompt)
            if result:
                return {"result": result, "source": f"gemini/{GEMINI_MODEL}"}
        raise HTTPException(
            status_code=503,
            detail=(
                "No local AI model found. "
                "Install Ollama from https://ollama.ai, then run: "
                "ollama pull qwen3:8b"
            ),
        )

    async def _ollama_generate(prompt: str, model: str) -> Optional[str]:
        """Call Ollama and return stripped text, or None on failure."""
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                r = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model":      model,
                        "prompt":     prompt,
                        "system":     system_prompt,
                        "stream":     False,
                        "think":      False,
                        "keep_alive": 0,
                        # Lower temperature for Kazakh reduces language drift
                        **({"options": {"temperature": 0.7}} if is_kazakh else {}),
                    },
                )
            if r.status_code == 200:
                raw = r.json().get("response", "").strip()
                return re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            log.warning(f"Model '{model}' returned HTTP {r.status_code}")
        except Exception as e:
            log.warning(f"Model '{model}' failed: {type(e).__name__}: {e}")
        return None

    # Try each model in preference order; fall back on failure.
    # Kazakh uses models[:1] (set above) — no retry allowed because keep_alive:0
    # unloads the model immediately, making a second call require a full reload
    # which starves ACE-Step of GPU memory.
    last_error = None

    for model in models:
        log.info(f"AI generate: model={model} language='{lang}'")
        result = await _ollama_generate(lyrics_prompt, model)
        if result is None:
            last_error = f"Model '{model}' failed or returned empty"
            continue

        if is_kazakh:
            score = _kazakh_score(result)
            log.info(f"Kazakh score for model '{model}': {score:.3f} ({len(result)} chars)")
            if score < 0.03:
                log.warning(f"Model '{model}' score={score:.3f} — low but returning (no retry to avoid GPU OOM)")

        log.info(f"AI response from model '{model}' ({len(result)} chars)")
        return {"result": result, "source": f"ollama/{model}"}

    # All models failed
    log.error(f"All {len(models)} models failed. Last error: {last_error}")
    raise HTTPException(status_code=502, detail=f"All AI models failed. Last error: {last_error}")

# ─── Audio Analysis (librosa) ─────────────────────────────────
def _run_analysis(audio_path: Path) -> dict:
    """
    Synchronous librosa analysis — runs in a thread pool.
    Analyzes the full audio for accurate cross-variant comparisons.
    """
    import librosa
    import numpy as np

    log.info(f"Analyzing: {audio_path.name}")
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)

    # BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(round(float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo), 1))

    # Key detection via chroma + Krumhansl-Schmuckler profiles
    chroma      = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1).tolist()

    major_profile = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
    minor_profile = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

    best_key, best_corr, best_mode = 'C', -999, 'major'
    for i in range(12):
        rotated = chroma_mean[i:] + chroma_mean[:i]
        corr_maj = sum(a*b for a,b in zip(rotated, major_profile))
        if corr_maj > best_corr:
            best_corr, best_key, best_mode = corr_maj, notes[i], 'major'
        corr_min = sum(a*b for a,b in zip(rotated, minor_profile))
        if corr_min > best_corr:
            best_corr, best_key, best_mode = corr_min, notes[i], 'minor'
    key = f"{best_key} {best_mode}"

    # Energy: 90th-percentile RMS normalised to a typical loud-track reference (~0.08).
    # More stable than mean×scalar across tracks with different loudness envelopes.
    rms     = librosa.feature.rms(y=y)[0]
    rms_p90 = float(np.percentile(rms, 90))
    energy_norm = float(min(1.0, rms_p90 / 0.08))

    # Spectral bands
    stft   = np.abs(librosa.stft(y))
    freqs  = librosa.fft_frequencies(sr=sr)
    bass_e   = float(np.mean(stft[(freqs >= 20)   & (freqs < 250)]))
    mid_e    = float(np.mean(stft[(freqs >= 250)  & (freqs < 4000)]))
    treble_e = float(np.mean(stft[(freqs >= 4000) & (freqs < 20000)]))
    total    = bass_e + mid_e + treble_e + 1e-8

    # Mood — mutually exclusive elif chain; thresholds chosen to avoid border ambiguity.
    valence = 0.65 if best_mode == 'major' else 0.35
    if energy_norm >= 0.65 and bpm >= 120:
        mood = "Energetic"
    elif energy_norm >= 0.65 and bpm < 120:
        mood = "Intense"
    elif energy_norm < 0.35 and bpm < 90:
        mood = "Calm"
    elif energy_norm < 0.35 and best_mode == 'minor':
        mood = "Melancholic"
    elif bpm >= 130 and valence > 0.5:
        mood = "Upbeat"
    elif valence >= 0.6:
        mood = "Happy"
    elif valence < 0.4:
        mood = "Dark"
    else:
        mood = "Balanced"

    result = {
        "bpm":      bpm,
        "key":      key,
        "energy":   round(energy_norm, 3),
        "mood":     mood,
        "spectrum": {
            "bass":   round((bass_e   / total) * 100, 1),
            "mid":    round((mid_e    / total) * 100, 1),
            "treble": round((treble_e / total) * 100, 1),
        },
        "duration": round(float(librosa.get_duration(y=y, sr=sr)), 1),
        "analyzed": True,
    }
    log.info(f"Analysis done: BPM={bpm} Key={key} Mood={mood}")
    return result

@app.get("/analyze/{task_id}")
async def analyze_track(task_id: str):
    """Analyze a track with librosa and return audio features."""
    if task_id in analysis_cache:
        return analysis_cache[task_id]

    audio_file = audio_file_map.get(task_id)
    if not audio_file:
        audio_file = find_audio_file(task_id)
    if not audio_file or not Path(audio_file).exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = await loop.run_in_executor(pool, _run_analysis, Path(audio_file))
    except Exception as e:
        log.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    _cache_set(analysis_cache, task_id, result)
    # Save analysis to database
    save_analysis_to_db(task_id, result)
    return result

# ─── Stems Export ─────────────────────────────────────────────
def _run_stems(audio_path: Path) -> bytes:
    """Separate audio into vocals + instrumental stems and return as ZIP bytes.

    Uses BSRoformer via audio-separator (best quality) with htdemucs_ft fallback.
    Runs synchronously — call via run_in_executor.
    """
    import shutil
    import numpy as np
    import soundfile as sf

    tmp_dir = Path(tempfile.mkdtemp(prefix="maz_stems_"))
    vocals_path       = tmp_dir / "vocals.wav"
    instrumental_path = tmp_dir / "instrumental.wav"
    separated = False

    # 1. Try BSRoformer (audio-separator)
    try:
        from audio_separator.separator import Separator
        sep_dir = tmp_dir / "sep"
        sep_dir.mkdir()
        sep = Separator(output_dir=str(sep_dir), output_format="WAV", log_level=30)
        sep.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
        out_files = sep.separate(str(audio_path))
        voc_files = [sep_dir / Path(f).name for f in out_files if "(Vocals)" in f]
        ins_files = [sep_dir / Path(f).name for f in out_files if "(Instrumental)" in f]
        if not voc_files: voc_files = sorted(sep_dir.glob("*(Vocals)*.wav"))
        if not ins_files: ins_files = sorted(sep_dir.glob("*(Instrumental)*.wav"))
        if voc_files and ins_files:
            shutil.copy(str(voc_files[0]), str(vocals_path))
            shutil.copy(str(ins_files[0]), str(instrumental_path))
            separated = True
            log.info("[stems] BSRoformer separation complete")
    except Exception as e:
        log.warning(f"[stems] BSRoformer failed: {e}; trying htdemucs_ft")

    # 2. Fallback: htdemucs_ft
    if not separated:
        try:
            import torch
            import librosa
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            model    = get_model("htdemucs_ft")
            model.eval()
            model_sr = model.samplerate
            device   = "cuda" if torch.cuda.is_available() else "cpu"
            wav_np, _ = librosa.load(str(audio_path), sr=model_sr, mono=False)
            if wav_np.ndim == 1:
                wav_np = np.stack([wav_np, wav_np])
            wav_tensor = torch.from_numpy(wav_np).float().unsqueeze(0)
            with torch.no_grad():
                sources = apply_model(model, wav_tensor, device=device, progress=False)[0]
            vocals_idx   = model.sources.index("vocals")
            no_vocals_np = sum(
                sources[i].cpu().numpy() for i, s in enumerate(model.sources) if s != "vocals"
            )
            sf.write(str(vocals_path),       sources[vocals_idx].cpu().numpy().T, model_sr)
            sf.write(str(instrumental_path), no_vocals_np.T,                model_sr)
            separated = True
            log.info("[stems] htdemucs_ft separation complete")
        except Exception as e:
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
            raise RuntimeError(f"Stem separation failed: {e}") from e

    # Bundle stems into a ZIP in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(str(vocals_path),       "vocals.wav")
        zf.write(str(instrumental_path), "instrumental.wav")
    shutil.rmtree(str(tmp_dir), ignore_errors=True)
    buf.seek(0)
    return buf.read()


@app.get("/stems/{task_id}")
async def get_stems(task_id: str):
    """Separate a track into vocals + instrumental and return as a ZIP download."""
    audio_file = _resolve_audio(task_id)
    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio file not found")

    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            zip_bytes = await loop.run_in_executor(pool, _run_stems, Path(audio_file))
    except Exception as e:
        log.error(f"Stems export failed for {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=maz_{task_id[:8]}_stems.zip",
            "Content-Length": str(len(zip_bytes)),
        },
    )


# ─── History & Statistics API ─────────────────────────────────
@app.get("/history")
async def get_history(
    limit: int = 50,
    offset: int = 0,
    favorites_only: bool = False,
    voice_sections_only: bool = False,
    sort: str = "newest",
    all_users: bool = False,
    user: dict = Depends(get_current_user),
):
    """Get generated tracks. Users see their own; admins may pass all_users=1."""
    clauses = []
    params: list = []

    # Scope to current user (include legacy tracks with no owner) unless admin requesting all
    if not (user["role"] == "admin" and all_users):
        clauses.append("(t.user_id = ? OR t.user_id IS NULL)")
        params.append(user["id"])

    if favorites_only:
        clauses.append("t.favorited = 1")
    elif voice_sections_only:
        clauses.append("t.voice_sections_path IS NOT NULL AND t.voice_sections_path != ''")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    _sort_map = {
        "newest":   "t.created_at DESC",
        "oldest":   "t.created_at ASC",
        "duration": "t.duration DESC NULLS LAST",
        "bpm":      "a.bpm DESC NULLS LAST",
        "energy":   "a.energy DESC NULLS LAST",
    }
    order_by = _sort_map.get(sort, "t.created_at DESC")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"""
            SELECT t.*,
                   a.bpm, a.key, a.mood, a.energy,
                   a.bass, a.mid, a.treble
            FROM tracks t
            LEFT JOIN analysis a ON a.track_id = t.id
            {where}
            ORDER BY {order_by}
            LIMIT ? OFFSET ?
        """, (*params, limit, offset)).fetchall()

    # Filter out rows whose audio file no longer exists on disk
    valid_rows = []
    for r in rows:
        p = r["audio_path"]
        if p and Path(p).exists():
            valid_rows.append(dict(r))

    return {
        "tracks": valid_rows,
        "total":  len(valid_rows),
        "limit":  limit,
        "offset": offset,
    }


@app.get("/history/bulk-download")
async def bulk_download(job_ids: str, _: dict = Depends(get_current_user)):
    """Stream a ZIP archive containing the selected tracks' audio files."""
    import io, zipfile as zf_mod
    from fastapi.responses import StreamingResponse as _SR
    ids = [j.strip() for j in job_ids.split(",") if j.strip()]
    buf = io.BytesIO()
    with zf_mod.ZipFile(buf, "w", zf_mod.ZIP_DEFLATED) as zf:
        with sqlite3.connect(DB_PATH) as conn:
            for job_id in ids:
                row = conn.execute(
                    "SELECT audio_path, voice_sections_path, prompt FROM tracks WHERE job_id=?",
                    (job_id,),
                ).fetchone()
                if not row:
                    continue
                path = row[1] or row[0]
                if path and os.path.exists(path):
                    safe = (row[2] or job_id)[:40].replace("/", "_").replace("\\", "_")
                    ext = Path(path).suffix or ".wav"
                    zf.write(path, f"{safe}_{job_id[:8]}{ext}")
    buf.seek(0)
    return _SR(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="maz-export.zip"'},
    )

@app.patch("/history/{job_id}/favorite")
async def toggle_favorite(job_id: str, user: dict = Depends(get_current_user)):
    """Toggle the favorited flag on a track."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT favorited, user_id FROM tracks WHERE job_id=?", (job_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Track not found")
        if user["role"] != "admin" and row[1] is not None and row[1] != user["id"]:
            raise HTTPException(status_code=403, detail="Not your track")
        new_val = 0 if row[0] else 1
        conn.execute("UPDATE tracks SET favorited=? WHERE job_id=?", (new_val, job_id))
        conn.commit()
    return {"job_id": job_id, "favorited": bool(new_val)}

@app.get("/history/prompts")
async def get_prompt_history(limit: int = 30, _: dict = Depends(get_current_user)):
    """Return the most recent unique prompts and lyrics for the studio history dropdowns."""
    with sqlite3.connect(DB_PATH) as conn:
        # Deduplicate by prompt text, keep most recent occurrence
        rows = conn.execute("""
            SELECT prompt, lyrics, MAX(created_at) as last_used
            FROM tracks
            WHERE prompt IS NOT NULL AND prompt != ''
            GROUP BY prompt
            ORDER BY last_used DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return {
        "prompts": [
            {"prompt": r[0], "lyrics": r[1], "last_used": r[2]}
            for r in rows
        ]
    }


@app.get("/history/stats")
async def get_stats(_: dict = Depends(get_current_user)):
    """Aggregate statistics for the diploma results chapter."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        total = conn.execute("SELECT COUNT(*) as n FROM tracks").fetchone()["n"]

        avg_bpm = conn.execute(
            "SELECT ROUND(AVG(bpm),1) as v FROM analysis"
        ).fetchone()["v"]

        top_moods = conn.execute("""
            SELECT mood, COUNT(*) as n
            FROM analysis WHERE mood IS NOT NULL
            GROUP BY mood ORDER BY n DESC LIMIT 5
        """).fetchall()

        top_keys = conn.execute("""
            SELECT key, COUNT(*) as n
            FROM analysis WHERE key IS NOT NULL
            GROUP BY key ORDER BY n DESC LIMIT 5
        """).fetchall()

        avg_duration = conn.execute(
            "SELECT ROUND(AVG(duration),1) as v FROM tracks"
        ).fetchone()["v"]

        avg_steps = conn.execute(
            "SELECT ROUND(AVG(steps),1) as v FROM tracks"
        ).fetchone()["v"]

        avg_energy = conn.execute(
            "SELECT ROUND(AVG(energy),3) as v FROM analysis"
        ).fetchone()["v"]

        recent_7d = conn.execute("""
            SELECT COUNT(*) as n FROM tracks
            WHERE created_at >= datetime('now', '-7 days')
        """).fetchone()["n"]

    return {
        "total_generated":  total,
        "recent_7_days":    recent_7d,
        "avg_bpm":          avg_bpm,
        "avg_duration":     avg_duration,
        "avg_steps":        avg_steps,
        "avg_energy":       avg_energy,
        "top_moods":        [dict(r) for r in top_moods],
        "top_keys":         [dict(r) for r in top_keys],
    }

@app.get("/history/export")
async def export_history(
    format: str = "csv",
    favorites_only: bool = False,
    job_ids: Optional[str] = None,
    search: Optional[str] = None,
    _: dict = Depends(get_current_user),
):
    """Export track history as CSV or JSON with optional filtering."""
    import io, csv as _csv
    from datetime import date as _date
    from fastapi.responses import Response

    conditions: list[str] = []
    params: list = []

    if favorites_only:
        conditions.append("t.favorited = 1")

    if job_ids:
        id_list = [j.strip() for j in job_ids.split(",") if j.strip()]
        if id_list:
            placeholders = ",".join("?" * len(id_list))
            conditions.append(f"t.job_id IN ({placeholders})")
            params.extend(id_list)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute(f"""
            SELECT t.job_id, t.title, t.prompt, t.lyrics,
                   t.duration, t.steps, t.guidance, t.seed,
                   t.bpm_input, t.key_input, t.favorited,
                   t.voice_sections_path, t.created_at,
                   a.bpm, a.key, a.mood, a.energy, a.bass, a.mid, a.treble
            FROM tracks t
            LEFT JOIN analysis a ON a.track_id = t.id
            {where}
            ORDER BY t.created_at DESC
        """, params).fetchall())

    # Optional text search (applied after DB fetch to reuse the query)
    if search:
        q = search.lower()
        rows = [r for r in rows if q in (r["prompt"] or "").lower()
                                or q in (r["title"] or "").lower()]

    today = _date.today().isoformat()
    n = len(rows)

    if format == "json":
        records = [
            {
                "job_id":               r["job_id"],
                "title":                r["title"] or "",
                "prompt":               r["prompt"] or "",
                "lyrics":               r["lyrics"] or "",
                "duration_s":           r["duration"],
                "steps":                r["steps"],
                "guidance":             r["guidance"],
                "seed":                 r["seed"],
                "bpm_input":            r["bpm_input"],
                "key_input":            r["key_input"] or "",
                "favorited":            bool(r["favorited"]),
                "has_voice_conversion": bool(r["voice_sections_path"]),
                "audio_url":            f"/audio/{r['job_id']}",
                "created_at":           r["created_at"],
                "detected_bpm":         round(r["bpm"], 1) if r["bpm"] else None,
                "detected_key":         r["key"] or None,
                "mood":                 r["mood"] or None,
                "energy":               round(r["energy"], 3) if r["energy"] else None,
                "bass_pct":             round(r["bass"], 1) if r["bass"] else None,
                "mid_pct":              round(r["mid"], 1) if r["mid"] else None,
                "treble_pct":           round(r["treble"], 1) if r["treble"] else None,
            }
            for r in rows
        ]
        body = json.dumps(
            {"total": n, "exported_at": today, "tracks": records},
            ensure_ascii=False, indent=2,
        ).encode("utf-8")
        return Response(
            content=body,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="maz_export_{today}.json"'},
        )

    # ── CSV ──────────────────────────────────────────────────────
    output = io.StringIO()
    writer = _csv.writer(output)
    writer.writerow([
        "job_id", "title", "prompt", "lyrics",
        "duration_s", "steps", "guidance", "seed",
        "bpm_input", "key_input", "favorited", "has_voice_conversion",
        "audio_url", "created_at",
        "detected_bpm", "detected_key", "mood", "energy",
        "bass_pct", "mid_pct", "treble_pct",
    ])
    for r in rows:
        writer.writerow([
            r["job_id"],
            r["title"] or "",
            r["prompt"] or "",
            (r["lyrics"] or "").replace("\n", " | "),   # flatten newlines for cell readability
            r["duration"],
            r["steps"],
            r["guidance"],
            r["seed"],
            r["bpm_input"] or "",
            r["key_input"] or "",
            1 if r["favorited"] else 0,
            1 if r["voice_sections_path"] else 0,
            f"/audio/{r['job_id']}",
            r["created_at"],
            round(r["bpm"], 1) if r["bpm"] else "",
            r["key"] or "",
            r["mood"] or "",
            round(r["energy"], 3) if r["energy"] else "",
            round(r["bass"], 1) if r["bass"] else "",
            round(r["mid"], 1) if r["mid"] else "",
            round(r["treble"], 1) if r["treble"] else "",
        ])

    csv_bytes = output.getvalue().encode("utf-8-sig")   # utf-8-sig for Excel compatibility
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="maz_export_{today}.csv"'},
    )

# ─── Voice Profiles ───────────────────────────────────────────
# In-memory registry: voice_id → {name, type}  ("svc" | "rvc")
voice_profile_registry: Dict[str, dict] = {}

# JSON index file that persists the registry across server restarts
VOICE_PROFILES_INDEX = VOICE_PROFILES_DIR / "profiles.json"

def _load_voice_index():
    """Load voice profile names from the JSON index on disk into the in-memory registry.
    Handles both legacy string format (plain name) and current dict format ({name, type})."""
    if not VOICE_PROFILES_INDEX.exists():
        return
    try:
        import json as _json
        data = _json.loads(VOICE_PROFILES_INDEX.read_text(encoding="utf-8"))
        for voice_id, val in data.items():
            if isinstance(val, str):
                # Legacy format: string name → treat as SVC
                wav = VOICE_PROFILES_DIR / f"{voice_id}.wav"
                if wav.exists():
                    voice_profile_registry[voice_id] = {"name": val, "type": "svc"}
            elif isinstance(val, dict):
                vtype = val.get("type", "svc")
                if vtype == "rvc":
                    pth = RVC_MODELS_DIR / f"{voice_id}.pth"
                    if pth.exists():
                        voice_profile_registry[voice_id] = val
                else:
                    wav = VOICE_PROFILES_DIR / f"{voice_id}.wav"
                    if wav.exists():
                        voice_profile_registry[voice_id] = val
        log.info(f"[voice] Loaded {len(voice_profile_registry)} voice profile(s) from index")
    except Exception as e:
        log.warning(f"[voice] Could not load voice index: {e}")

def _save_voice_index():
    """Persist the in-memory registry to the JSON index file."""
    try:
        import json as _json
        VOICE_PROFILES_INDEX.write_text(
            _json.dumps(voice_profile_registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        log.warning(f"[voice] Could not save voice index: {e}")


def _prepare_voice_reference(input_tmp: Path, output_path: Path) -> bool:
    """Isolate vocals from an uploaded clip, denoise, and save as the voice profile.

    Priority:
      1. audio-separator BSRoformer (best quality, SDR ~13)
      2. htdemucs_ft fallback
      3. raw audio fallback (normalised only)
    Applies two-pass noisereduce after separation so Seed-VC learns a clean timbre.
    Returns True if separation succeeded, False if raw-audio fallback was used.
    Blocking — call via asyncio.to_thread().
    """
    import tempfile
    import numpy as np
    import librosa
    import soundfile as sf

    separated = False
    vocals_np = None
    out_sr    = 44100

    # ── 1. Try audio-separator (BSRoformer) ───────────────────────────────────
    try:
        from audio_separator.separator import Separator

        log.info("[voice] Separating vocals with BSRoformer…")
        sep_dir = Path(tempfile.mkdtemp(prefix="maz_ref_sep_"))
        sep = Separator(output_dir=str(sep_dir), output_format="WAV", log_level=30)
        sep.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
        out_files = sep.separate(str(input_tmp))

        voc_files = [Path(f) for f in out_files if "(Vocals)" in Path(f).name]
        if not voc_files:
            voc_files = sorted(sep_dir.glob("*(Vocals)*.wav"))

        if voc_files:
            vocals_np, out_sr = sf.read(str(voc_files[0]))  # (samples, ch)
            if vocals_np.ndim == 1:
                vocals_np = np.stack([vocals_np, vocals_np])
            else:
                vocals_np = vocals_np.T                      # (ch, samples)
            separated = True
            log.info("[voice] BSRoformer separation complete")
        else:
            log.warning("[voice] BSRoformer produced no Vocals file, falling back")

    except Exception as e:
        log.warning(f"[voice] audio-separator failed: {e}; trying htdemucs_ft")

    # ── 2. Fallback: htdemucs_ft ──────────────────────────────────────────────
    if not separated:
        try:
            import torch
            from demucs.pretrained import get_model
            from demucs.apply import apply_model

            log.info("[voice] Separating vocals with htdemucs_ft…")
            model    = get_model("htdemucs_ft")
            model.eval()
            out_sr   = model.samplerate

            wav_np, _ = librosa.load(str(input_tmp), sr=out_sr, mono=False)
            if wav_np.ndim == 1:
                wav_np = np.stack([wav_np, wav_np])

            wav_tensor = torch.from_numpy(wav_np).float().unsqueeze(0)
            _demucs_device = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.no_grad():
                sources = apply_model(model, wav_tensor, device=_demucs_device, progress=True)[0]

            vocals_idx = model.sources.index("vocals")
            vocals_np  = sources[vocals_idx].numpy()   # (ch, samples)
            separated  = True
            log.info("[voice] htdemucs_ft separation complete")

        except Exception as e:
            log.warning(f"[voice] htdemucs_ft failed: {e}; using raw audio")

    # ── 3. Raw audio fallback ─────────────────────────────────────────────────
    if not separated:
        wav_np, out_sr = librosa.load(str(input_tmp), sr=None, mono=False)
        if wav_np.ndim == 1:
            wav_np = np.stack([wav_np, wav_np])
        vocals_np = wav_np

    # ── Two-pass noisereduce — clean before Seed-VC learns the timbre ─────────
    try:
        import noisereduce as nr
        cleaned = []
        for ch in vocals_np:
            ch = nr.reduce_noise(y=ch, sr=out_sr, stationary=True,
                                 prop_decrease=0.85, n_fft=2048, n_jobs=1)
            ch = nr.reduce_noise(y=ch, sr=out_sr, stationary=False,
                                 prop_decrease=0.75, n_fft=2048, n_jobs=1)
            cleaned.append(ch.astype(np.float32))
        vocals_np = np.stack(cleaned)
        log.info("[voice] noisereduce applied to reference")
    except Exception as nr_err:
        log.warning(f"[voice] noisereduce skipped: {nr_err}")

    # Normalise to 0.95 peak and save as 44100 Hz stereo
    peak = float(np.max(np.abs(vocals_np)))
    if peak > 0:
        vocals_np = vocals_np * (0.95 / peak)

    sf.write(str(output_path), vocals_np.T, out_sr)
    log.info(f"[voice] Reference saved: {out_sr} Hz stereo, separated={separated}")
    return separated


@app.post("/voice/register")
async def voice_register(
    file:       UploadFile           = File(...),
    name:       str                  = Form(...),
    type:       str                  = Form("svc"),
    index_file: Optional[UploadFile] = File(None),
    _: dict = Depends(get_current_user),
):
    """Accept an SVC voice recording or an RVC model file and register it as a voice profile.

    SVC (type="svc"): file is a short audio recording — vocals are isolated and saved as .wav.
    RVC (type="rvc"): file is a trained .pth model; index_file is an optional .index for FAISS retrieval.
    """
    if type == "rvc":
        voice_id   = uuid.uuid4().hex[:12]
        model_path = RVC_MODELS_DIR / f"{voice_id}.pth"
        model_path.write_bytes(await file.read())
        if index_file and index_file.filename:
            idx_path = RVC_MODELS_DIR / f"{voice_id}.index"
            idx_path.write_bytes(await index_file.read())
        display_name = name.strip() or f"Voice {voice_id[:6]}"
        voice_profile_registry[voice_id] = {"name": display_name, "type": "rvc"}
        _save_voice_index()
        log.info(f"[voice] Registered RVC model '{display_name}' → {model_path.name}")
        return {"voice_id": voice_id, "name": display_name, "type": "rvc", "cleaned": False}

    # SVC path — existing vocal isolation flow
    suffix = Path(file.filename or "voice.wav").suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        input_tmp = Path(tmp.name)

    voice_id    = uuid.uuid4().hex[:12]
    output_path = VOICE_PROFILES_DIR / f"{voice_id}.wav"

    try:
        used_demucs = await asyncio.to_thread(_prepare_voice_reference, input_tmp, output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process voice: {e}")
    finally:
        input_tmp.unlink(missing_ok=True)

    display_name = name.strip() or f"Voice {voice_id[:6]}"
    voice_profile_registry[voice_id] = {"name": display_name, "type": "svc"}
    _save_voice_index()
    log.info(f"[voice] Registered SVC '{display_name}' (demucs={used_demucs}) → {output_path.name}")
    return {"voice_id": voice_id, "name": display_name, "type": "svc", "cleaned": used_demucs}


@app.get("/voice/profiles")
async def voice_list_profiles(_: dict = Depends(get_current_user)):
    """List all saved voice profiles (SVC and RVC)."""
    profiles = []
    for vid, vdata in voice_profile_registry.items():
        if isinstance(vdata, dict):
            pname = vdata.get("name", vid)
            vtype = vdata.get("type", "svc")
        else:
            pname = str(vdata)
            vtype = "svc"
        check = RVC_MODELS_DIR / f"{vid}.pth" if vtype == "rvc" else VOICE_PROFILES_DIR / f"{vid}.wav"
        if check.exists():
            profiles.append({"voice_id": vid, "name": pname, "type": vtype})
    return {"profiles": profiles}


@app.delete("/voice/profiles/{voice_id}")
async def voice_delete_profile(voice_id: str, _: dict = Depends(get_current_user)):
    """Delete a voice profile (SVC wav or RVC model files)."""
    profile = voice_profile_registry.get(voice_id, {})
    vtype   = profile.get("type", "svc") if isinstance(profile, dict) else "svc"
    if vtype == "rvc":
        (RVC_MODELS_DIR / f"{voice_id}.pth").unlink(missing_ok=True)
        (RVC_MODELS_DIR / f"{voice_id}.index").unlink(missing_ok=True)
    else:
        (VOICE_PROFILES_DIR / f"{voice_id}.wav").unlink(missing_ok=True)
    voice_profile_registry.pop(voice_id, None)
    _save_voice_index()
    return {"deleted": voice_id}


@app.delete("/history/{job_id}")
async def delete_track(job_id: str, user: dict = Depends(get_current_user)):
    """Delete a track from history. Users can only delete their own tracks."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id, user_id FROM tracks WHERE job_id=?", (job_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Track not found")
        if user["role"] != "admin" and row[1] is not None and row[1] != user["id"]:
            raise HTTPException(status_code=403, detail="Not your track")
        conn.execute("DELETE FROM analysis WHERE track_id=?", (row[0],))
        conn.execute("DELETE FROM tracks WHERE job_id=?", (job_id,))
        conn.commit()
    return {"deleted": job_id}


class TitleUpdate(BaseModel):
    title: str = Field(..., max_length=200)

@app.patch("/history/{job_id}/title")
async def rename_track(job_id: str, body: TitleUpdate, user: dict = Depends(get_current_user)):
    """Update the user-visible title of a track."""
    title = body.title.strip()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id, user_id FROM tracks WHERE job_id=?", (job_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Track not found")
        if user["role"] != "admin" and row[1] is not None and row[1] != user["id"]:
            raise HTTPException(status_code=403, detail="Not your track")
        conn.execute(
            "UPDATE tracks SET title=? WHERE job_id=?",
            (title or None, job_id),  # empty string → NULL (falls back to prompt)
        )
        conn.commit()
    return {"job_id": job_id, "title": title}


# ─── Voice Sections ───────────────────────────────────────────
class VoiceSectionItem(BaseModel):
    label:    str
    voice_id: Optional[str] = None

class VoiceSectionsRequest(BaseModel):
    job_id:   str
    sections: List[VoiceSectionItem]

async def _run_sections_pipeline(job_id: str, audio_path: Path,
                                  audio_dur: float, sections_data: list):
    """Launch voice_sections_worker.py and relay progress over WebSocket."""
    map_key    = f"sections_{job_id}"
    output_wav = VOICE_SECTIONS_DIR / f"{job_id}.wav"
    await broadcast({"type": "voice_sections_progress", "job_id": job_id,
                     "status": "processing", "msg": "Starting voice sections…"})
    import tempfile as _tempfile
    _fd, _tmp = _tempfile.mkstemp(suffix=".json", prefix="maz_vsec_")
    os.close(_fd)
    payload_path = Path(_tmp)
    proc = None
    try:
        step = audio_dur / max(len(sections_data), 1)
        payload_path.write_text(json.dumps([
            {"label": s["label"], "start": round(i * step, 3),
             "end": round((i + 1) * step, 3), "voice_ref": s.get("voice_ref")}
            for i, s in enumerate(sections_data)
        ]), encoding="utf-8")
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(VOICE_SECTIONS_WORKER),
            str(audio_path), str(payload_path), str(output_wav),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        process_guard.register(proc)
        # Drain stderr concurrently so a full stderr buffer can't deadlock the process.
        stderr_task = asyncio.create_task(proc.stderr.read())
        async for line_bytes in proc.stdout:
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if line:
                log.info(f"[sections] {line}")
                await broadcast({"type": "voice_sections_progress", "job_id": job_id,
                                 "status": "processing", "msg": line})
        await asyncio.wait_for(proc.wait(), timeout=VOICE_PIPELINE_TIMEOUT)
        stderr_b = await stderr_task
        if proc.returncode == 0 and output_wav.exists():
            _cache_set(audio_file_map, map_key, output_wav)
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute("UPDATE tracks SET voice_sections_path=? WHERE job_id=?",
                                 (str(output_wav), job_id))
                    conn.commit()
            except Exception as e:
                log.error(f"[sections] DB update failed: {e}")
            await broadcast({"type": "voice_sections_done", "job_id": job_id,
                             "audio_url": f"/audio/{map_key}"})
            log.info(f"[sections] Done → {output_wav.name}")
        else:
            err = stderr_b.decode("utf-8", errors="replace")[-600:] if stderr_b else "unknown error"
            log.error(f"[sections] Worker failed (rc={proc.returncode}):\n{err}")
            # Surface the first meaningful line of the error to the user
            user_err = next((ln.strip() for ln in err.splitlines() if ln.strip()), "Voice sections processing failed")
            await broadcast({"type": "voice_sections_failed", "job_id": job_id,
                             "error": user_err})
    except asyncio.TimeoutError:
        await broadcast({"type": "voice_sections_failed", "job_id": job_id,
                         "error": "Voice sections timed out (30 min)"})
    except Exception as e:
        log.error(f"[sections] Exception: {e}")
        await broadcast({"type": "voice_sections_failed", "job_id": job_id, "error": str(e)})
    finally:
        if proc is not None:
            process_guard.unregister(proc)
        payload_path.unlink(missing_ok=True)

@app.post("/voice/sections")
async def apply_voice_sections(req: VoiceSectionsRequest, _: dict = Depends(get_current_user)):
    """Apply per-section voice conversion (SVC) to a generated track."""
    if not req.sections:
        raise HTTPException(status_code=400, detail="No sections provided")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT audio_path, duration FROM tracks WHERE job_id=?", (req.job_id,)
            ).fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Track not found")

    audio_path = Path(row[0])
    # Also check audio_file_map (may have been updated by voice pipeline)
    for key in (req.job_id, f"sections_{req.job_id}"):
        if key in audio_file_map and audio_file_map[key].exists():
            audio_path = audio_file_map[key]; break
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found on disk")

    try:
        import soundfile as _sf
        audio_dur = _sf.info(str(audio_path)).duration
    except Exception:
        audio_dur = float(row[1]) if row[1] else 60.0

    sections_data = []
    for s in req.sections:
        voice_ref = None
        if s.voice_id:
            profile = voice_profile_registry.get(s.voice_id, {})
            if isinstance(profile, dict) and profile.get("type") != "rvc":
                p = VOICE_PROFILES_DIR / f"{s.voice_id}.wav"
                if p.exists():
                    voice_ref = str(p)
        sections_data.append({"label": s.label, "voice_ref": voice_ref})

    if not any(s.get("voice_ref") for s in sections_data):
        raise HTTPException(status_code=400,
                            detail="No valid SVC voice profiles assigned. RVC voices are not supported for sections.")
    asyncio.create_task(_run_sections_pipeline(req.job_id, audio_path, audio_dur, sections_data))
    return {"status": "processing", "job_id": req.job_id}


# ─── Voice Training ───────────────────────────────────────────
@app.post("/voice/train")
async def start_voice_training(
    files:       List[UploadFile] = File(...),
    name:        str              = Form("My Trained Voice"),
    epochs:      int              = Form(80),
    sample_rate: int              = Form(40000),
    batch_size:  int              = Form(16),
    _: dict = Depends(get_current_user),
):
    """Upload audio samples and start an in-app voice training job."""
    job_id      = uuid.uuid4().hex[:12]
    dataset_dir = RVC_DATASETS_DIR / job_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        suffix = Path(f.filename or "audio.wav").suffix.lower() or ".wav"
        dest   = dataset_dir / (uuid.uuid4().hex[:8] + suffix)
        dest.write_bytes(await f.read())

    model_name = (name.strip() or f"voice_{job_id[:6]}")
    out_dir    = RVC_TRAINING_OUTPUT / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    job: dict = {
        "job_id":     job_id,
        "name":       model_name,
        "status":     "running",
        "progress":   0.0,
        "stage":      "init",
        "msg":        "Starting…",
        "losses":     [],
        "voice_id":   None,
        "started_at": datetime.now().isoformat(),
    }
    _cache_set(voice_training_jobs, job_id, job)
    asyncio.create_task(_run_training(job_id, dataset_dir, out_dir, model_name,
                                      epochs, sample_rate, batch_size))
    log.info(f"[train] Job {job_id} started — '{model_name}' epochs={epochs}")
    return {"job_id": job_id, "name": model_name, "status": "running"}


async def _run_training(job_id: str, dataset_dir: Path, out_dir: Path,
                        model_name: str, epochs: int, sr: int, batch: int):
    """Stream training worker stdout as WebSocket progress broadcasts."""
    import shutil
    job  = voice_training_jobs[job_id]
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(VOICE_TRAIN_WORKER),
            str(dataset_dir), str(out_dir), model_name,
            str(epochs), str(sr), str(batch),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        voice_training_procs[job_id] = proc
        process_guard.register(proc)

        train_start = asyncio.get_running_loop().time()
        while True:
            try:
                line_bytes = await asyncio.wait_for(proc.stdout.readline(), timeout=300)
            except asyncio.TimeoutError:
                proc.kill()
                raise asyncio.TimeoutError("Training stalled — no output for 5 minutes")
            if not line_bytes:
                break
            if asyncio.get_running_loop().time() - train_start > VOICE_TRAIN_TIMEOUT:
                proc.kill()
                raise asyncio.TimeoutError(f"Training exceeded {VOICE_TRAIN_TIMEOUT}s limit")
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            stage    = data.get("stage", job["stage"])
            progress = float(data.get("progress", job["progress"]))
            msg      = data.get("msg", "")
            job.update({"stage": stage, "progress": progress, "msg": msg})
            if "loss" in data:
                loss_val = float(data["loss"])
                if math.isfinite(loss_val):
                    job["losses"].append(round(loss_val, 6))
                if len(job["losses"]) > 200:
                    job["losses"] = job["losses"][-200:]

            await broadcast({
                "type": "training_progress",
                "job_id": job_id,
                "stage": stage,
                "progress": progress,
                "msg": msg,
                "losses": job["losses"][-50:],
            })

            if stage == "done":
                # Copy trained files into the voice profile directories
                ref_wav = out_dir / f"{model_name}_ref.wav"
                pth_src = out_dir / f"{model_name}.pth"
                idx_src = out_dir / f"{model_name}.index"
                voice_id = job_id  # reuse job_id as the voice profile id

                if ref_wav.exists():
                    shutil.copy(str(ref_wav), str(VOICE_PROFILES_DIR / f"{voice_id}.wav"))
                if pth_src.exists():
                    shutil.copy(str(pth_src), str(RVC_MODELS_DIR / f"{voice_id}.pth"))
                if idx_src.exists():
                    shutil.copy(str(idx_src), str(RVC_MODELS_DIR / f"{voice_id}.index"))

                # Use "rvc" when a .pth model was produced; fall back to "svc" only if
                # solely a WAV reference was created (no .pth output).
                trained_type = "rvc" if pth_src.exists() else "svc"
                display = job["name"]
                voice_profile_registry[voice_id] = {"name": display, "type": trained_type, "trained": True}
                _save_voice_index()
                job["voice_id"] = voice_id
                job["status"]   = "completed"
                log.info(f"[train] {job_id} complete — voice_id={voice_id}")
                break

            if stage == "error":
                job["status"] = "failed"
                job["msg"]    = msg
                break

        await proc.wait()
        if job["status"] == "running":
            if proc.returncode == 0:
                job["status"] = "completed"
            else:
                stderr = await proc.stderr.read()
                job["status"] = "failed"
                job["msg"]    = stderr.decode(errors="replace")[-500:] or "Worker exited with error"

    except Exception as e:
        job["status"] = "failed"
        job["msg"]    = str(e)
        log.error(f"[train] {job_id} exception: {e}")

    finally:
        voice_training_procs.pop(job_id, None)
        if proc is not None:
            process_guard.unregister(proc)
        await broadcast({
            "type":     "training_progress",
            "job_id":   job_id,
            "stage":    job.get("stage", "done"),
            "progress": job.get("progress", 1.0),
            "msg":      job.get("msg", ""),
            "losses":   job.get("losses", [])[-50:],
            "status":   job["status"],
            "voice_id": job.get("voice_id"),
        })


@app.get("/voice/train")
async def list_training_jobs(_: dict = Depends(get_current_user)):
    """List all training jobs (current session)."""
    return {"jobs": list(voice_training_jobs.values())}


@app.get("/voice/train/{job_id}")
async def get_training_status(job_id: str, _: dict = Depends(get_current_user)):
    """Get the current state of a training job."""
    job = voice_training_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


@app.delete("/voice/train/{job_id}")
async def cancel_training(job_id: str, _: dict = Depends(get_current_user)):
    """Terminate a running training job."""
    job = voice_training_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    proc = voice_training_procs.get(job_id)
    if proc and proc.returncode is None:
        try:
            proc.terminate()
        except Exception:
            pass
    job["status"] = "cancelled"
    log.info(f"[train] {job_id} cancelled by user")
    return {"cancelled": job_id}


# ─── Generation Presets ────────────────────────────────────────

@app.get("/presets")
async def list_presets(user: dict = Depends(get_current_user)):
    """Return presets for the current user, newest first."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, name, params, created_at FROM presets WHERE user_id=? ORDER BY id DESC",
            (user["id"],),
        ).fetchall()
    return [
        {"id": r["id"], "name": r["name"], "created_at": r["created_at"],
         **json.loads(r["params"])}
        for r in rows
    ]


@app.post("/presets", status_code=201)
async def save_preset(req: PresetSave, user: dict = Depends(get_current_user)):
    """Save the current Studio parameters as a named preset for the current user."""
    params = req.model_dump(exclude={"name"})
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO presets (name, params, user_id) VALUES (?, ?, ?)",
            (req.name.strip(), json.dumps(params), user["id"]),
        )
        preset_id = cur.lastrowid
        conn.commit()
    log.info(f"Preset saved: '{req.name}' id={preset_id} user={user['username']}")
    return {"id": preset_id, "name": req.name, **params}


@app.delete("/presets/{preset_id}", status_code=200)
async def delete_preset(preset_id: int, user: dict = Depends(get_current_user)):
    """Delete a saved preset. Users can only delete their own presets."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT user_id FROM presets WHERE id=?", (preset_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Preset not found")
        if user["role"] != "admin" and row[0] != user["id"]:
            raise HTTPException(status_code=403, detail="Not your preset")
        conn.execute("DELETE FROM presets WHERE id = ?", (preset_id,))
        conn.commit()
    log.info(f"Preset deleted: id={preset_id}")
    return {"deleted": preset_id}