# MAZ — AI Music Generation Studio

MAZ is a self-hosted web app for AI music generation. It pairs the
[ACE-Step 1.5](https://github.com/ace-step/ACE-Step) foundation model with a
clean browser UI and a queue-driven REST API, then layers voice conversion,
stem separation, mastering, and library management on top.

Generate a song from a text prompt. Convert vocals with your own voice. Export
stems, master to −14 LUFS, download as WAV/MP3/FLAC. Everything runs locally.

---

## Features

- **Text-to-music** — full songs (vocals + instruments) from a prompt and
  optional lyrics, powered by the ACE-Step 1.5 LLM-planner + DiT pipeline.
- **Voice conversion** — Seed-VC and RVC pipelines for swapping the singing
  voice on a generated track.
- **Section-level voicing** — apply different voice profiles to chorus, verse,
  bridge, etc. independently.
- **Stems export** — separate any track into vocals + instrumental
  (BSRoformer → htdemucs_ft fallback) and download as a ZIP.
- **Mastering** — RMS→LUFS normalization to −14 LUFS with a true-peak limiter
  at −1 dBFS.
- **Multi-format download** — WAV (lossless), MP3 (V2), FLAC (lossless).
- **Audio analysis** — automatic BPM, key, mood, and energy detection
  (librosa-based, cached in SQLite).
- **Real-time progress** — WebSocket-driven UI updates for generation,
  voice processing, and training.
- **Voice profile training** — train your own RVC voice models from short
  audio samples directly in the UI.
- **Admin panic endpoint** — reset stuck state and kill orphan worker
  processes without rebooting.
- **Hardware aware** — works on CUDA, ROCm, MPS, MLX, and CPU; auto-detects
  via the underlying ACE-Step engine.
- **No build step** — the frontend is vanilla JS + CSS, no bundler required.

---

## Architecture

```
Browser (vanilla JS, WebSocket)
        │
        ▼
MAZ web server  (FastAPI, port 3000)
        │
        ├─ HTTP polling ──▶ ACE-Step API server (port 8001)
        │                    └─ LLM planner → DiT → VAE → audio
        │
        ├─ subprocess ────▶ voice_pipeline_worker.py    (Seed-VC / RVC)
        ├─ subprocess ────▶ voice_sections_worker.py    (per-section voicing)
        └─ subprocess ────▶ rvc_training_worker.py      (voice model training)
```

MAZ is the orchestrator. ACE-Step is a separate process — start it first.

---

## Prerequisites

- **Python 3.11 or 3.12**
- **[ACE-Step 1.5](https://github.com/ace-step/ACE-Step)** running on port 8001
- **ffmpeg** on `PATH` (used for MP3 conversion and mastering)
- **GPU** strongly recommended (4 GB VRAM minimum for ACE-Step turbo;
  CPU-only is supported but slow)

For voice features:
- **[Seed-VC](https://github.com/Plachtaa/seed-vc)** — required for SVC voice
  conversion. Place at `../seed-vc/` relative to the soundforge directory, or
  edit the path in `voice_pipeline_worker.py`.
- **RVC model files** — `.pth` files dropped into `outputs/voice_profiles/rvc_models/`.

---

## Installation

```bash
# 1. Clone this repo
git clone https://github.com/<your-username>/maz.git
cd maz

# 2. Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env — at minimum, change ADMIN_PASSWORD
```

Set up ACE-Step separately following its
[installation guide](https://github.com/ace-step/ACE-Step#install).

---

## Running

Two terminals:

**Terminal 1 — ACE-Step engine** (start first):
```bash
cd <path-to-ace-step>
uv run acestep-api          # listens on :8001
```

**Terminal 2 — MAZ server**:
```bash
cd maz
uvicorn server:app --host 0.0.0.0 --port 3000 --env-file .env
```

Open `http://localhost:3000` in your browser.

### Offline mode

ACE-Step can run without internet after the first successful boot:
```bash
uv run --offline acestep-api
```

---

## Configuration

All settings live in `.env`. See [`.env.example`](.env.example) for the
complete list.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ACESTEP_API` | `http://localhost:8001` | URL of the ACE-Step API |
| `ADMIN_PASSWORD` | `admin1234` | Guards `/admin/*` endpoints — **change this** |
| `MAX_PER_DAY` | `10` | Daily generation limit per client IP |
| `ACESTEP_AUDIO_DIR` | (auto) | Where ACE-Step writes audio files |

---

## API

REST endpoints exposed by MAZ:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/generate` | Queue a music generation job |
| `GET` | `/job/{id}` | Poll a job's status |
| `DELETE` | `/job/{id}` | Cancel a queued or active job |
| `GET` | `/audio/{task_id}` | Stream the generated audio |
| `GET` | `/download/{task_id}?format=wav\|mp3\|flac` | Download in chosen format |
| `GET` | `/stems/{task_id}` | Download vocals + instrumental as ZIP |
| `GET` | `/master/{task_id}` | Download a −14 LUFS mastered version |
| `GET` | `/analyze/{task_id}` | BPM / key / mood / energy |
| `POST` | `/voice/sections` | Apply per-section voice conversion |
| `POST` | `/voice/train` | Train an RVC voice model |
| `WS` | `/ws` | Real-time progress events |

Admin endpoints (require `?password=…`):

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/admin?password=…` | Admin dashboard |
| `DELETE` | `/admin/active?password=…` | Force-clear a stuck active job |
| `DELETE` | `/admin/queue?password=…` | Drop the entire queue |
| `POST` | `/admin/panic?password=…` | Kill child workers + orphans, reset state |

---

## Recovery

If generation gets stuck and a server restart doesn't help, hit the panic endpoint:

```bash
curl.exe -X POST "http://localhost:3000/admin/panic?password=admin1234"
```

This kills every tracked child worker, sweeps for orphan worker processes
left from a prior crash, drops the active job, and clears the queue. Returns
a JSON summary of what was cleaned up.

---

## Project layout

```
soundforge/
├── server.py                    # FastAPI app, queue worker, all endpoints
├── process_guard.py             # Subprocess lifecycle / orphan cleanup
├── voice_pipeline_worker.py     # Seed-VC / RVC voice conversion
├── voice_sections_worker.py     # Per-section voicing
├── rvc_training_worker.py       # RVC voice model training
├── static/                      # Vanilla JS frontend (no build step)
│   ├── app.js
│   ├── index.html
│   └── style.css
├── requirements.txt
├── .env.example
└── README.md
```

State written at runtime (gitignored):
- `maz.db` — SQLite database with track history and audio analysis
- `outputs/` — generated audio, voice profiles, voice sections, training data

---

## Tech stack

**Backend:** FastAPI · uvicorn · httpx · WebSockets · SQLite · psutil
**Audio:** librosa · soundfile · numpy · ffmpeg · audio-separator · demucs
**ML:** ACE-Step 1.5 (text-to-music) · Seed-VC + RVC (voice conversion) ·
torch
**Frontend:** vanilla JS · CSS · Web Notifications API · Web Audio API

---

## Acknowledgments

MAZ is a thin orchestration layer around several outstanding open-source
projects — none of the heavy lifting is original:

- [**ACE-Step**](https://github.com/ace-step/ACE-Step) — text-to-music
  foundation model
- [**Seed-VC**](https://github.com/Plachtaa/seed-vc) — singing voice
  conversion
- [**RVC**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
  — retrieval-based voice conversion
- [**Demucs**](https://github.com/facebookresearch/demucs) — stem separation
- [**audio-separator**](https://github.com/karaokenerds/python-audio-separator)
  — BSRoformer / MDX23 wrappers
- [**librosa**](https://librosa.org/) — audio analysis

---

## License

[MIT](./LICENSE) — do whatever you want with the MAZ code itself.

The underlying models and dependencies have their own licenses (mostly MIT or
similar permissive terms). Check the individual project pages before using
generated content commercially.
