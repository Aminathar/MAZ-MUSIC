"""Voice pipeline worker — run as subprocess from maz/server.py.

Usage:
    python voice_pipeline_worker.py <generated_audio> <voice_reference> <output_path>

Pipeline:
    1. Stem separation — BSRoformer (fallback: htdemucs_ft) splits song into
       vocals + instrumental track.
    2. HPF (40 Hz) — remove sub-bass rumble while preserving chest warmth/resonance.
    3. Seed-VC voice conversion — SEEDVC_STEPS steps (default 30), cfg-rate SEEDVC_CFG_RATE (default 0.7).
    4. Voice restoration chain (all scipy, no heavy neural models):
         a. Parametric EQ — 40 Hz low-cut, −1.5 dB boxiness at 350 Hz,
            −1.5 dB gentle metallic notch at 1.5 kHz, +2 dB presence at 3 kHz,
            +2 dB air at 10 kHz
         b. Frequency-selective de-esser (6–9 kHz, 3:1, −20 dB threshold)
         c. Subtle room reverb (6% wet) — just enough to seat the voice in the mix
    5. RMS-loudness-matched mix — vocals sit 6 dB over BGM by RMS, not peak.

Exits 0 and prints "OK: <output_path>" on success.
Exits 1 and prints error to stderr on failure.
"""
import os
import sys
import shutil
import traceback
import tempfile
import subprocess
from pathlib import Path

# ─── Args ────────────────────────────────────────────────────────────────────
if len(sys.argv) < 4:
    print("Usage: voice_pipeline_worker.py <generated_audio> <voice_ref> <output>",
          file=sys.stderr)
    sys.exit(1)

generated_audio = Path(sys.argv[1]).resolve()
voice_reference = Path(sys.argv[2]).resolve()
output_path     = Path(sys.argv[3]).resolve()

SEED_VC_DIR     = Path(__file__).parent.parent / "seed-vc"
DIFFUSION_STEPS = int(os.environ.get("SEEDVC_STEPS",    "30"))
FP16            = os.environ.get("SEEDVC_FP16",         "True")
F0_CONDITION    = os.environ.get("SEEDVC_F0",           "True")
# cfg-rate: how closely to follow reference voice character (0=prosody-free, 1=full imitation)
SEEDVC_CFG_RATE = os.environ.get("SEEDVC_CFG_RATE",     "0.7")
VOICE_TYPE      = os.environ.get("VOICE_TYPE",          "svc").lower()

# ─── Stem cache ──────────────────────────────────────────────────────────────
import hashlib as _hashlib

_STEMS_CACHE = Path(__file__).parent / "stems_cache"
_STEMS_CACHE.mkdir(exist_ok=True)

def _cache_key(p: Path) -> str:
    s = p.stat()
    return _hashlib.md5(f"{p}:{s.st_size}:{s.st_mtime}".encode()).hexdigest()

def _cached_stems(audio: Path):
    k = _cache_key(audio)
    v, i = _STEMS_CACHE / f"{k}_v.wav", _STEMS_CACHE / f"{k}_i.wav"
    return (v, i) if v.exists() and i.exists() else (None, None)

def _save_stems(audio: Path, voc: Path, ins: Path) -> None:
    k = _cache_key(audio)
    shutil.copy(str(voc), str(_STEMS_CACHE / f"{k}_v.wav"))
    shutil.copy(str(ins), str(_STEMS_CACHE / f"{k}_i.wav"))

# ─── Validate ────────────────────────────────────────────────────────────────
if not generated_audio.exists():
    print(f"Generated audio not found: {generated_audio}", file=sys.stderr)
    sys.exit(1)

if VOICE_TYPE != "rvc":
    if not voice_reference.exists():
        print(f"Voice reference not found: {voice_reference}", file=sys.stderr)
        sys.exit(1)
    if not SEED_VC_DIR.exists():
        print(f"Seed-VC not found at {SEED_VC_DIR}.", file=sys.stderr)
        sys.exit(1)

# ─── RVC branch ──────────────────────────────────────────────────────────────
if VOICE_TYPE == "rvc":
    RVC_MODEL_PATH = os.environ.get("RVC_MODEL_PATH", str(voice_reference))
    RVC_INDEX_PATH = os.environ.get("RVC_INDEX_PATH", "")

    try:
        from rvc_python.infer import RVCInference
    except ImportError:
        print(
            "rvc-python is not installed.\n"
            "Install it with:  pip install rvc-python\n"
            "Then restart the server.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import numpy as np
        import soundfile as sf
        import librosa

        rvc_tmp = Path(tempfile.mkdtemp(prefix="maz_rvc_"))
        vocals_path_r    = rvc_tmp / "vocals.wav"
        no_vocals_path_r = rvc_tmp / "no_vocals.wav"
        sep_dir_r        = rvc_tmp / "sep"
        sep_dir_r.mkdir()

        # ── Stem separation (cache → BSRoformer → Demucs fallback) ──────────
        _cv, _ci = _cached_stems(generated_audio)
        if _cv:
            shutil.copy(str(_cv), str(vocals_path_r))
            shutil.copy(str(_ci), str(no_vocals_path_r))
            separated = True
            print("  Using cached stems", flush=True)
        else:
            separated = False

        try:
            from audio_separator.separator import Separator
            print("[RVC 1/3] Separating stems with BSRoformer…", flush=True)
            sep = Separator(output_dir=str(sep_dir_r), output_format="WAV", log_level=30)
            sep.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
            out_files = sep.separate(str(generated_audio))
            voc_files = [sep_dir_r / Path(f).name for f in out_files if "(Vocals)" in f]
            ins_files = [sep_dir_r / Path(f).name for f in out_files if "(Instrumental)" in f]
            if not voc_files: voc_files = sorted(sep_dir_r.glob("*(Vocals)*.wav"))
            if not ins_files: ins_files = sorted(sep_dir_r.glob("*(Instrumental)*.wav"))
            if voc_files and ins_files:
                shutil.copy(str(voc_files[0]), str(vocals_path_r))
                shutil.copy(str(ins_files[0]), str(no_vocals_path_r))
                separated = True
                print("  BSRoformer done", flush=True)
        except Exception as e:
            print(f"  BSRoformer failed ({e}), trying htdemucs_ft", flush=True)

        if not separated:
            import torch
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            print("[RVC 1/3] Separating stems with htdemucs_ft…", flush=True)
            model = get_model("htdemucs_ft")
            model.eval()
            model_sr = model.samplerate
            wav_np, _ = librosa.load(str(generated_audio), sr=model_sr, mono=False)
            if wav_np.ndim == 1:
                wav_np = np.stack([wav_np, wav_np])
            wav_tensor = torch.from_numpy(wav_np).float().unsqueeze(0)
            _demucs_device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  demucs device: {_demucs_device}", flush=True)
            with torch.no_grad():
                sources = apply_model(model, wav_tensor, device=_demucs_device, progress=True)[0]
            vocals_idx = model.sources.index("vocals")
            no_vocals_np = sum(
                sources[i].cpu().numpy()
                for i, s in enumerate(model.sources) if s != "vocals"
            )
            sf.write(str(vocals_path_r), sources[vocals_idx].cpu().numpy().T, model_sr)
            sf.write(str(no_vocals_path_r), no_vocals_np.T, model_sr)
            print("  htdemucs_ft done", flush=True)

        if not _cv:
            _save_stems(generated_audio, vocals_path_r, no_vocals_path_r)

        # ── HPF ──────────────────────────────────────────────────────────────
        try:
            from scipy.signal import butter, sosfilt
            voc_np, voc_sr = sf.read(str(vocals_path_r))
            if voc_np.ndim == 1:
                voc_np = np.stack([voc_np, voc_np])
            else:
                voc_np = voc_np.T
            sos = butter(4, 40.0 / (voc_sr / 2), btype="high", output="sos")
            voc_np = np.stack([sosfilt(sos, ch).astype(np.float32) for ch in voc_np])
            sf.write(str(vocals_path_r), voc_np.T, voc_sr)
        except Exception as e:
            print(f"  HPF skipped: {e}", flush=True)

        # ── RVC inference ─────────────────────────────────────────────────────
        print("[RVC 2/3] Running RVC inference…", flush=True)
        rvc_out = rvc_tmp / "rvc_vocals.wav"

        import torch as _torch
        device = "cuda:0" if _torch.cuda.is_available() else "cpu"
        rvc = RVCInference(device=device)
        try:
            rvc.load_model(RVC_MODEL_PATH, RVC_INDEX_PATH if RVC_INDEX_PATH else None)
        except TypeError:
            rvc.load_model(RVC_MODEL_PATH)
            if RVC_INDEX_PATH:
                try:
                    rvc.set_index(RVC_INDEX_PATH)
                except Exception:
                    pass
        rvc.infer_file(str(vocals_path_r), str(rvc_out))
        print(f"  RVC inference done → {rvc_out.name}", flush=True)

        # ── RMS-matched mix ───────────────────────────────────────────────────
        print("[RVC 3/3] Mixing vocals with instrumental…", flush=True)
        bgm,    sr_b = librosa.load(str(no_vocals_path_r), sr=None, mono=False)
        vocals, _    = librosa.load(str(rvc_out), sr=sr_b, mono=False)
        if bgm.ndim == 1: bgm = np.stack([bgm, bgm])
        if vocals.ndim == 1: vocals = np.stack([vocals, vocals])

        max_len = max(bgm.shape[-1], vocals.shape[-1])
        if bgm.shape[-1]    < max_len: bgm    = np.pad(bgm,    ((0,0),(0, max_len - bgm.shape[-1])))
        if vocals.shape[-1] < max_len: vocals = np.pad(vocals, ((0,0),(0, max_len - vocals.shape[-1])))

        bgm_peak = float(np.max(np.abs(bgm)))
        if bgm_peak > 0:
            bgm = bgm * (0.65 / bgm_peak)
        rms_bgm = float(np.sqrt(np.mean(bgm ** 2)) + 1e-9)
        rms_v   = float(np.sqrt(np.mean(vocals ** 2)) + 1e-9)
        vocals  = vocals * (rms_bgm * (10 ** (6.0 / 20)) / rms_v)
        v_peak  = float(np.max(np.abs(vocals)))
        if v_peak > 0.92:
            vocals = vocals * (0.92 / v_peak)

        mixed = bgm + vocals
        peak  = float(np.max(np.abs(mixed)))
        if peak > 0.98:
            mixed = mixed * (0.98 / peak)

        sf.write(str(output_path), mixed.T, sr_b)
        print(f"OK: {output_path}", flush=True)

    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            shutil.rmtree(rvc_tmp, ignore_errors=True)
        except Exception:
            pass

    sys.exit(0)


# ─── Step 1: Stem separation ──────────────────────────────────────────────────
try:
    import numpy as np
    import soundfile as sf
    import librosa

    tmp_dir = Path(tempfile.mkdtemp(prefix="maz_voice_"))
    vocals_path    = tmp_dir / "vocals.wav"
    no_vocals_path = tmp_dir / "no_vocals.wav"
    sep_dir        = tmp_dir / "sep"
    sep_dir.mkdir()

    # ── Stem cache check ─────────────────────────────────────────────────────
    _svc_cv, _svc_ci = _cached_stems(generated_audio)
    if _svc_cv:
        shutil.copy(str(_svc_cv), str(vocals_path))
        shutil.copy(str(_svc_ci), str(no_vocals_path))
        separated = True
        print("  Using cached stems", flush=True)
    else:
        separated = False

    # ── Try audio-separator (BSRoformer — best quality) ──────────────────────
    try:
        from audio_separator.separator import Separator

        print(f"[1/4] Separating stems with BSRoformer (downloads model on first run)…",
              flush=True)
        sep = Separator(
            output_dir=str(sep_dir),
            output_format="WAV",
            log_level=30,  # WARNING only
        )
        sep.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
        out_files = sep.separate(str(generated_audio))

        # audio-separator returns bare filenames — resolve against output dir
        voc_files = [sep_dir / Path(f).name for f in out_files if "(Vocals)" in f]
        ins_files = [sep_dir / Path(f).name for f in out_files if "(Instrumental)" in f]

        if not voc_files:
            voc_files = sorted(sep_dir.glob("*(Vocals)*.wav"))
        if not ins_files:
            ins_files = sorted(sep_dir.glob("*(Instrumental)*.wav"))

        if voc_files and ins_files:
            shutil.copy(str(voc_files[0]), str(vocals_path))
            shutil.copy(str(ins_files[0]), str(no_vocals_path))
            separated = True
            print(f"  BSRoformer separation complete", flush=True)
        else:
            print("  BSRoformer output files not found, falling back to Demucs",
                  flush=True)

    except Exception as e:
        print(f"  audio-separator failed ({e}), falling back to htdemucs_ft", flush=True)

    # ── Fallback: htdemucs_ft ─────────────────────────────────────────────────
    if not separated:
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        print(f"[1/4] Separating stems with htdemucs_ft…", flush=True)
        model = get_model("htdemucs_ft")
        model.eval()
        model_sr = model.samplerate
        _demucs_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  demucs device: {_demucs_device}", flush=True)

        wav_np, _ = librosa.load(str(generated_audio), sr=model_sr, mono=False)
        if wav_np.ndim == 1:
            wav_np = np.stack([wav_np, wav_np])

        wav_tensor = torch.from_numpy(wav_np).float().unsqueeze(0)
        with torch.no_grad():
            sources = apply_model(model, wav_tensor, device=_demucs_device, progress=True)[0]

        vocals_idx   = model.sources.index("vocals")
        vocals_np    = sources[vocals_idx].cpu().numpy()
        no_vocals_np = sum(
            sources[i].cpu().numpy()
            for i, s in enumerate(model.sources) if s != "vocals"
        )
        sf.write(str(vocals_path),    vocals_np.T,    model_sr)
        sf.write(str(no_vocals_path), no_vocals_np.T, model_sr)
        print(f"  htdemucs_ft separation complete", flush=True)

    if not _svc_cv:
        _save_stems(generated_audio, vocals_path, no_vocals_path)

    # ── High-pass filter: remove sub-40 Hz rumble before Seed-VC ─────────────
    # Pre-denoise removed: BSRoformer separation already yields clean vocals,
    # and Seed-VC's diffusion process handles residual noise internally.
    try:
        from scipy.signal import butter, sosfilt
        vocals_np, voc_sr = sf.read(str(vocals_path))
        if vocals_np.ndim == 1:
            vocals_np = np.stack([vocals_np, vocals_np])
        else:
            vocals_np = vocals_np.T
        sos = butter(4, 40.0 / (voc_sr / 2), btype='high', output='sos')
        hpf_channels = [sosfilt(sos, ch).astype(np.float32) for ch in vocals_np]
        vocals_np = np.stack(hpf_channels)
        sf.write(str(vocals_path), vocals_np.T, voc_sr)
        print("  HPF (40 Hz) applied", flush=True)
    except Exception as e:
        print(f"  HPF skipped: {e}", flush=True)

    print(f"  vocals    -> {vocals_path}", flush=True)
    print(f"  no_vocals -> {no_vocals_path}", flush=True)

except Exception:
    print(traceback.format_exc(), file=sys.stderr)
    sys.exit(1)

# ─── Step 2: Seed-VC voice conversion ────────────────────────────────────────
try:
    vc_out_dir = tmp_dir / "vc_out"
    vc_out_dir.mkdir(exist_ok=True)
    print(f"[2/4] Running Seed-VC ({DIFFUSION_STEPS} steps)…", flush=True)

    seed_vc_script = SEED_VC_DIR / "inference.py"
    cmd = [
        sys.executable, str(seed_vc_script),
        "--source",             str(vocals_path),
        "--target",             str(voice_reference),
        "--output",             str(vc_out_dir),
        "--diffusion-steps",    str(DIFFUSION_STEPS),
        "--length-adjust",      "1.0",
        "--inference-cfg-rate", SEEDVC_CFG_RATE,
        "--fp16",               FP16,
        "--f0-condition",       F0_CONDITION,
    ]
    # auto-f0-adjust is intentionally OFF: it shifts the source melody pitch to match
    # the reference voice's median range, which corrupts the intended musical notes.

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SEED_VC_DIR))
    if result.returncode != 0:
        print(f"Seed-VC failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    candidates = sorted(vc_out_dir.glob("vc_*.wav"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(
            [p for p in tmp_dir.rglob("*.wav")
             if p != vocals_path and p != no_vocals_path],
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
    if not candidates:
        print("Seed-VC produced no output file.", file=sys.stderr)
        sys.exit(1)

    converted_vocals = candidates[0]
    print(f"  converted -> {converted_vocals}", flush=True)

except Exception:
    print(traceback.format_exc(), file=sys.stderr)
    sys.exit(1)

# ─── Step 3: Voice restoration chain ────────────────────────────────────────
try:
    import numpy as np
    import soundfile as sf
    from scipy.signal import lfilter, butter, sosfilt, fftconvolve

    print("[3/4] Restoring converted vocals…", flush=True)

    wav_np, conv_sr = sf.read(str(converted_vocals))
    if wav_np.ndim == 1:
        wav_np = np.stack([wav_np, wav_np])
    else:
        wav_np = wav_np.T
    sr = conv_sr

    # ── Biquad filter helpers (Audio EQ Cookbook, R. Bristow-Johnson) ─────────
    # Proper minimum-phase IIR filters — unlike Butterworth + bandpass
    # subtraction, these have no phase ringing or inter-band artifacts.
    def _bq(x, b, a):
        return np.array([lfilter(b, a, ch).astype(np.float32) for ch in x])

    def _hp(f, Q=0.707):
        w = 2 * np.pi * f / sr; c = np.cos(w); alpha = np.sin(w) / (2 * Q)
        return (np.array([(1+c)/2, -(1+c), (1+c)/2]),
                np.array([1+alpha, -2*c,   1-alpha]))

    def _peak(f, gain_db, Q):
        A = 10 ** (gain_db / 40); w = 2 * np.pi * f / sr
        c = np.cos(w); alpha = np.sin(w) / (2 * Q)
        return (np.array([1+alpha*A, -2*c, 1-alpha*A]),
                np.array([1+alpha/A, -2*c, 1-alpha/A]))

    def _hshelf(f, gain_db, S=1.0):
        A = 10 ** (gain_db / 40); w = 2 * np.pi * f / sr; c = np.cos(w)
        alpha = np.sin(w) / 2 * np.sqrt((A + 1/A) * (1/S - 1) + 2)
        sq = 2 * np.sqrt(A) * alpha
        b = np.array([ A*((A+1)+(A-1)*c+sq), -2*A*((A-1)+(A+1)*c),  A*((A+1)+(A-1)*c-sq)])
        a = np.array([   (A+1)-(A-1)*c+sq,    2*  ((A-1)-(A+1)*c),    (A+1)-(A-1)*c-sq  ])
        return b / a[0], np.array([1, a[1]/a[0], a[2]/a[0]])

    # ── 3a. Parametric EQ ────────────────────────────────────────────────────
    #   80 Hz HP     — strip sub-bass rumble, preserve chest warmth
    #  350 Hz −1.5 dB Q1.5 — reduce boxy resonance without thinning the voice
    # 1500 Hz −1.5 dB Q1.5 — gentle metallic notch (was −3 dB; too much = lifeless)
    # 3000 Hz +2.0 dB shelf — restore presence and consonant clarity
    # 10kHz  +2.0 dB shelf — restore air and breathiness (emotion lives here too)
    try:
        wav_np = _bq(wav_np, *_hp(40))
        wav_np = _bq(wav_np, *_peak(350,  -1.5, 1.5))
        wav_np = _bq(wav_np, *_peak(1500, -1.5, 1.5))
        wav_np = _bq(wav_np, *_hshelf(3000, 2.0))
        if sr > 22000:
            wav_np = _bq(wav_np, *_hshelf(10000, 2.0))
        print("  parametric EQ applied", flush=True)
    except Exception as e:
        print(f"  EQ skipped: {e}", flush=True)

    # ── 3b. Frequency-selective de-esser (6–9 kHz) ───────────────────────────
    # Extracts the sibilant band, reduces its gain when envelope > threshold,
    # then re-adds to the dry signal. Only the sibilant component is touched.
    try:
        nyq = sr / 2
        SIB_LO, SIB_HI = 6000.0, min(9000.0, nyq * 0.95)
        if SIB_LO < nyq and SIB_HI > SIB_LO:
            sos_sib = butter(4, [SIB_LO/nyq, SIB_HI/nyq], btype='band', output='sos')
            sos_env = butter(1, 80/nyq, btype='low', output='sos')
            THRESH  = 10 ** (-20.0 / 20)
            RATIO   = 3.0
            de_essed = []
            for ch in wav_np:
                sib  = sosfilt(sos_sib, ch)
                rest = ch - sib
                rms  = np.sqrt(np.maximum(sosfilt(sos_env, sib ** 2), 0.0))
                over = rms / (THRESH + 1e-9)
                gain = np.where(over > 1.0, over ** (1.0 / RATIO - 1.0), 1.0)
                gain = np.clip(gain, 0.3, 1.0)
                de_essed.append((rest + sib * gain).astype(np.float32))
            wav_np = np.stack(de_essed)
            print("  de-esser applied (6–9 kHz, 3:1)", flush=True)
    except Exception as e:
        print(f"  de-esser skipped: {e}", flush=True)

    # ── 3c. Room reverb (7% wet, RT60 ≈ 0.38 s) ─────────────────────────────
    # Seed-VC output is bone-dry. A short room IR blends the voice into the
    # acoustic space of the music without adding audible reverb.
    try:
        DECAY_S  = 0.28
        PRE_DLY  = int(0.005 * sr)
        ir_len   = int(sr * DECAY_S)
        t        = np.linspace(0.0, DECAY_S, ir_len, dtype=np.float32)
        rng      = np.random.default_rng(42)
        ir       = rng.standard_normal(ir_len).astype(np.float32) * np.exp(-t / 0.055)
        ir      /= np.max(np.abs(ir)) + 1e-9
        ir_pad   = np.concatenate([np.zeros(PRE_DLY, dtype=np.float32), ir])
        WET      = 0.06
        rev_chs  = []
        for ch in wav_np:
            wet = fftconvolve(ch, ir_pad)[:len(ch)].astype(np.float32)
            rev_chs.append(ch * (1.0 - WET) + wet * WET)
        wav_np = np.stack(rev_chs)
        print(f"  room reverb applied (6% wet)", flush=True)
    except Exception as e:
        print(f"  reverb skipped: {e}", flush=True)

    peak = float(np.max(np.abs(wav_np)))
    if peak > 0.0:
        wav_np = wav_np * (0.90 / peak)

    post_path = tmp_dir / "post.wav"
    sf.write(str(post_path), wav_np.T, conv_sr)
    converted_vocals = post_path
    print("  restoration complete", flush=True)

except Exception:
    print(traceback.format_exc(), file=sys.stderr)
    sys.exit(1)

# ─── Step 4: RMS-matched mix ─────────────────────────────────────────────────
try:
    import numpy as np
    import soundfile as sf
    import librosa

    print("[4/4] Mixing vocals with instrumental…", flush=True)

    bgm,    sr   = librosa.load(str(no_vocals_path),   sr=None, mono=False)
    vocals, sr_v = librosa.load(str(converted_vocals), sr=sr,   mono=False)

    if bgm.ndim == 1:
        bgm = np.stack([bgm, bgm])
    if vocals.ndim == 1:
        vocals = np.stack([vocals, vocals])

    # Pad/trim to same length
    max_len = max(bgm.shape[-1], vocals.shape[-1])
    if bgm.shape[-1] < max_len:
        bgm    = np.pad(bgm,    ((0, 0), (0, max_len - bgm.shape[-1])))
    if vocals.shape[-1] < max_len:
        vocals = np.pad(vocals, ((0, 0), (0, max_len - vocals.shape[-1])))

    # RMS-based loudness matching — vocals should sit ~4 dB above the BGM
    # (industry standard for a lead vocal). Peak-based normalisation ignores
    # perceived loudness and causes vocals to either disappear or drown the mix.
    rms_bgm    = float(np.sqrt(np.mean(bgm ** 2))    + 1e-9)
    rms_vocals = float(np.sqrt(np.mean(vocals ** 2)) + 1e-9)

    bgm_peak = float(np.max(np.abs(bgm)))
    if bgm_peak > 0:
        bgm = bgm * (0.65 / bgm_peak)
        rms_bgm = float(np.sqrt(np.mean(bgm ** 2)) + 1e-9)

    TARGET_OVER_DB = 6.0   # dB vocals over BGM
    target_rms_v   = rms_bgm * (10 ** (TARGET_OVER_DB / 20))
    vocals         = vocals * (target_rms_v / rms_vocals)

    v_peak = float(np.max(np.abs(vocals)))
    if v_peak > 0.92:
        vocals = vocals * (0.92 / v_peak)

    mixed = bgm + vocals
    peak  = float(np.max(np.abs(mixed)))
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)

    sf.write(str(output_path), mixed.T, sr)
    print(f"OK: {output_path}", flush=True)

except Exception:
    print(traceback.format_exc(), file=sys.stderr)
    sys.exit(1)

finally:
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass
