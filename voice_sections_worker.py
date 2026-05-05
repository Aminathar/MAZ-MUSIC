"""Voice sections worker — one stem separation, per-section Seed-VC, stitch.

Usage:
    python voice_sections_worker.py <audio_path> <sections_json> <output_path>

sections_json is a JSON array:
    [{"label": "Verse 1", "start": 0.0, "end": 22.5, "voice_ref": "/abs/path.wav"}, ...]
    voice_ref=null means keep original vocals for that section.

Exits 0 and prints "OK: <path>" on success.
Exits 1 with traceback to stderr on failure.
"""
import os
import sys
import json
import shutil
import tempfile
import traceback
import subprocess
from pathlib import Path

if len(sys.argv) < 4:
    print("Usage: voice_sections_worker.py <audio> <sections_json> <output>", file=sys.stderr)
    sys.exit(1)

audio_path   = Path(sys.argv[1]).resolve()
_payload_arg = sys.argv[2]
# argv[2] is either a .json file path (new) or raw JSON (legacy)
if Path(_payload_arg).exists():
    sections = json.loads(Path(_payload_arg).read_text(encoding="utf-8"))
else:
    sections = json.loads(_payload_arg)
output_path  = Path(sys.argv[3]).resolve()

SEED_VC_DIR     = Path(__file__).parent.parent / "seed-vc"
DIFFUSION_STEPS = int(os.environ.get("SEEDVC_STEPS", "50"))
FP16            = os.environ.get("SEEDVC_FP16", "True")
F0_CONDITION    = os.environ.get("SEEDVC_F0", "True")

if not audio_path.exists():
    print(f"Audio not found: {audio_path}", file=sys.stderr)
    sys.exit(1)

tmp_dir = Path(tempfile.mkdtemp(prefix="maz_vsec_"))

try:
    import numpy as np
    import soundfile as sf
    import librosa
    from scipy.signal import butter, sosfilt

    def _hpf(arr, sr):
        sos = butter(4, 80.0 / (sr / 2), btype="high", output="sos")
        return np.stack([sosfilt(sos, ch).astype(np.float32) for ch in arr])

    # ── Step 1: Stem separation (once for entire track) ───────────────────────
    print("[1/3] Separating stems…", flush=True)
    vocals_path    = tmp_dir / "vocals.wav"
    no_vocals_path = tmp_dir / "no_vocals.wav"
    separated      = False

    try:
        from audio_separator.separator import Separator
        sep_dir = tmp_dir / "sep"
        sep_dir.mkdir()
        sep = Separator(output_dir=str(sep_dir), output_format="WAV", log_level=30)
        sep.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
        out_files = sep.separate(str(audio_path))
        voc_f = [Path(f) for f in out_files if "(Vocals)" in Path(f).name]
        ins_f = [Path(f) for f in out_files if "(Instrumental)" in Path(f).name]
        if not voc_f: voc_f = sorted(sep_dir.glob("*(Vocals)*.wav"))
        if not ins_f: ins_f = sorted(sep_dir.glob("*(Instrumental)*.wav"))
        if voc_f and ins_f:
            shutil.copy(str(voc_f[0]), str(vocals_path))
            shutil.copy(str(ins_f[0]), str(no_vocals_path))
            separated = True
            print("  BSRoformer done", flush=True)
    except Exception as e:
        print(f"  BSRoformer failed ({e}), trying htdemucs_ft", flush=True)

    if not separated:
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        model = get_model("htdemucs_ft")
        model.eval()
        wav_np, _ = librosa.load(str(audio_path), sr=model.samplerate, mono=False)
        if wav_np.ndim == 1:
            wav_np = np.stack([wav_np, wav_np])
        _demucs_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  demucs device: {_demucs_device}", flush=True)
        with torch.no_grad():
            sources = apply_model(model, torch.from_numpy(wav_np).float().unsqueeze(0),
                                  device=_demucs_device, progress=True)[0]
        vi = model.sources.index("vocals")
        no_v = sum(sources[i].numpy() for i, s in enumerate(model.sources) if s != "vocals")
        sf.write(str(vocals_path), sources[vi].numpy().T, model.samplerate)
        sf.write(str(no_vocals_path), no_v.T, model.samplerate)
        print("  htdemucs_ft done", flush=True)

    voc_np, voc_sr = sf.read(str(vocals_path))
    bgm_np, _      = sf.read(str(no_vocals_path))
    if voc_np.ndim == 1: voc_np = np.stack([voc_np, voc_np])
    else:                voc_np = voc_np.T                  # (ch, samples)
    if bgm_np.ndim == 1: bgm_np = np.stack([bgm_np, bgm_np])
    else:                bgm_np = bgm_np.T

    result_voc = voc_np.copy().astype(np.float32)

    # ── Step 2: Per-section Seed-VC conversion ────────────────────────────────
    assigned = [s for s in sections if s.get("voice_ref") and Path(s["voice_ref"]).exists()]
    print(f"[2/3] Processing {len(assigned)} assigned section(s) of {len(sections)}…", flush=True)
    if not assigned:
        raise RuntimeError("No sections have a valid voice reference file — nothing to convert.")

    converted_count = 0
    for si, sec in enumerate(sections):
        voice_ref = sec.get("voice_ref")
        if not voice_ref or not Path(voice_ref).exists():
            continue
        start_f = int(float(sec["start"]) * voc_sr)
        end_f   = min(int(float(sec["end"]) * voc_sr), result_voc.shape[-1])
        if start_f >= end_f:
            continue
        label = sec.get("label", f"sec{si + 1}")
        print(f"  [{si+1}/{len(sections)}] {label}: {sec['start']:.1f}s–{sec['end']:.1f}s", flush=True)

        seg = _hpf(result_voc[:, start_f:end_f], voc_sr)
        seg_in = tmp_dir / f"seg_{si}.wav"
        sf.write(str(seg_in), seg.T, voc_sr)

        vc_dir = tmp_dir / f"vc_{si}"
        vc_dir.mkdir()
        cmd = [sys.executable, str(SEED_VC_DIR / "inference.py"),
               "--source", str(seg_in), "--target", str(voice_ref),
               "--output", str(vc_dir), "--diffusion-steps", str(DIFFUSION_STEPS),
               "--length-adjust", "1.0", "--inference-cfg-rate", "0.7",
               "--fp16", FP16, "--f0-condition", F0_CONDITION]
        # auto-f0-adjust intentionally OFF — it shifts melody pitch to match reference
        # voice range, corrupting musical notes. Source pitch is preserved as-is.
        r = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace",
                           cwd=str(SEED_VC_DIR))
        if r.returncode != 0:
            err_snippet = (r.stderr or "")[-300:].strip()
            print(f"  Seed-VC failed for {label}: {err_snippet}", flush=True)
            continue

        cands = sorted(vc_dir.glob("vc_*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            print(f"  No output file for {label}, keeping original", flush=True)
            continue

        conv_np, conv_sr = sf.read(str(cands[0]))
        if conv_np.ndim == 1: conv_np = np.stack([conv_np, conv_np])
        else:                  conv_np = conv_np.T
        if conv_sr != voc_sr:
            conv_np = np.stack([librosa.resample(ch, orig_sr=conv_sr, target_sr=voc_sr)
                                 for ch in conv_np])

        orig_len = end_f - start_f
        if conv_np.shape[-1] > orig_len:
            conv_np = conv_np[:, :orig_len]
        elif conv_np.shape[-1] < orig_len:
            conv_np = np.pad(conv_np, ((0, 0), (0, orig_len - conv_np.shape[-1])))

        # 50ms crossfade at section edges to avoid clicks
        fade = min(int(0.05 * voc_sr), orig_len // 4)
        if fade > 0:
            conv_np[:, :fade]  *= np.linspace(0, 1, fade, dtype=np.float32)
            conv_np[:, -fade:] *= np.linspace(1, 0, fade, dtype=np.float32)

        result_voc[:, start_f:end_f] = conv_np.astype(np.float32)
        converted_count += 1
        print(f"  {label}: done", flush=True)

    if converted_count == 0:
        raise RuntimeError("Seed-VC failed for all assigned sections — check logs above.")

    # ── Step 3: RMS-matched mix ───────────────────────────────────────────────
    print("[3/3] Mixing…", flush=True)
    bgm_np = bgm_np.astype(np.float32)
    max_len = max(bgm_np.shape[-1], result_voc.shape[-1])
    if bgm_np.shape[-1]    < max_len: bgm_np    = np.pad(bgm_np,    ((0, 0), (0, max_len - bgm_np.shape[-1])))
    if result_voc.shape[-1] < max_len: result_voc = np.pad(result_voc, ((0, 0), (0, max_len - result_voc.shape[-1])))

    bgm_peak = float(np.max(np.abs(bgm_np)))
    if bgm_peak > 0:
        bgm_np = bgm_np * (0.65 / bgm_peak)
    rms_bgm = float(np.sqrt(np.mean(bgm_np ** 2)) + 1e-9)
    rms_v   = float(np.sqrt(np.mean(result_voc ** 2)) + 1e-9)
    result_voc = result_voc * (rms_bgm * (10 ** (4.0 / 20)) / rms_v)
    v_peak = float(np.max(np.abs(result_voc)))
    if v_peak > 0.88:
        result_voc = result_voc * (0.88 / v_peak)

    mixed = bgm_np + result_voc
    peak  = float(np.max(np.abs(mixed)))
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)

    sf.write(str(output_path), mixed.T, voc_sr)
    print(f"OK: {output_path}", flush=True)

except Exception:
    print(traceback.format_exc(), file=sys.stderr)
    sys.exit(1)

finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)

sys.exit(0)
