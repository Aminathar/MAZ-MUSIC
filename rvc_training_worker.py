"""
MAZ Voice Training Worker
=========================
Trains a speaker embedding model from multiple audio samples and produces a
high-quality voice reference WAV for use with Seed-VC voice conversion.

Reports JSON progress lines to stdout:
  {"stage": "preprocess", "progress": 0.45, "msg": "Processing file.wav…"}
  {"stage": "train",      "progress": 0.72, "msg": "Epoch 58/80", "loss": 0.123}
  {"stage": "done",       "progress": 1.0,  "msg": "Training complete"}
  {"stage": "error",      "progress": 0.0,  "msg": "description"}

Usage:
  python rvc_training_worker.py INPUT_DIR OUTPUT_DIR MODEL_NAME [epochs] [sr] [batch]
"""

import sys
import json
import tempfile
from pathlib import Path


def emit(stage: str, progress: float, msg: str = "", loss: float = None):
    d = {"stage": stage, "progress": round(float(progress), 4), "msg": msg}
    if loss is not None:
        d["loss"] = round(float(loss), 6)
    print(json.dumps(d), flush=True)


def main():
    if len(sys.argv) < 4:
        emit("error", 0, "Args: INPUT_DIR OUTPUT_DIR MODEL_NAME [epochs] [sr] [batch]")
        sys.exit(1)

    input_dir  = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    model_name = sys.argv[3]
    epochs     = int(sys.argv[4]) if len(sys.argv) > 4 else 150
    target_sr  = int(sys.argv[5]) if len(sys.argv) > 5 else 40000
    batch_size = int(sys.argv[6]) if len(sys.argv) > 6 else 16
    output_dir.mkdir(parents=True, exist_ok=True)

    emit("init", 0.0, "Initializing training pipeline…")

    try:
        import numpy as np
        import librosa
        import soundfile as sf
    except ImportError as e:
        emit("error", 0, f"Missing dependency: {e}. Run: pip install numpy librosa soundfile")
        sys.exit(1)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False
        emit("init", 0.05, "PyTorch not found — using lightweight embedding mode")

    # ── Collect audio files ──────────────────────────────────────
    EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    audio_files = [f for f in input_dir.rglob("*") if f.suffix.lower() in EXTS]
    if not audio_files:
        emit("error", 0, f"No audio files found in {input_dir}")
        sys.exit(1)
    emit("preprocess", 0.02, f"Found {len(audio_files)} audio file(s)")

    # ── Vocal isolation per file ─────────────────────────────────
    emit("preprocess", 0.05, "Isolating vocals from audio files…")
    clean_files = []
    for i, src in enumerate(audio_files):
        pct = (i + 1) / max(len(audio_files), 1)
        emit("preprocess", 0.05 + pct * 0.30, f"Separating {src.name}…")
        try:
            from audio_separator.separator import Separator
            sep_dir = Path(tempfile.mkdtemp(prefix="maz_sep_"))
            sep = Separator(output_dir=str(sep_dir), output_format="WAV", log_level=40)
            sep.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
            outs = sep.separate(str(src))
            voc = [Path(f) for f in outs if "(Vocals)" in Path(f).name]
            if not voc:
                voc = sorted(sep_dir.glob("*(Vocals)*.wav"))
            if voc:
                clean_files.append(voc[0])
                continue
        except Exception:
            pass
        clean_files.append(src)

    # ── Load and slice into 4-second segments ────────────────────
    emit("preprocess", 0.35, "Segmenting audio…")
    PROC_SR   = 22050           # 22 kHz captures full vocal detail up to 11 kHz
    CHUNK_LEN = int(4.0 * PROC_SR)   # 4 s — long enough to encode emotional phrasing
    HOP_LEN   = int(2.0 * PROC_SR)   # 50 % overlap
    MIN_LEN   = int(0.75 * PROC_SR)  # 0.75 s minimum

    segments = []
    segment_files = []   # parallel list: which clean_file each segment came from
    for i, fpath in enumerate(clean_files):
        pct = (i + 1) / max(len(clean_files), 1)
        emit("preprocess", 0.35 + pct * 0.55, f"Loading {fpath.name}…")
        try:
            y, _ = librosa.load(str(fpath), sr=PROC_SR, mono=True)
        except Exception:
            continue
        pk = float(np.max(np.abs(y)))
        if pk > 0:
            y = y / pk * 0.95
        for pos in range(0, len(y), HOP_LEN):
            seg = y[pos : pos + CHUNK_LEN]
            if len(seg) >= MIN_LEN:
                if len(seg) < CHUNK_LEN:
                    seg = np.pad(seg, (0, CHUNK_LEN - len(seg)))
                segments.append(seg.astype(np.float32))
                segment_files.append(fpath)

    if not segments:
        emit("error", 0, "No usable audio segments extracted. Check that files contain audio.")
        sys.exit(1)

    # Drop mostly-silent segments — they dilute the speaker model and hurt quality
    active_pairs = [(s, f) for s, f in zip(segments, segment_files)
                    if float(np.sqrt(np.mean(s ** 2))) > 0.005]
    if len(active_pairs) >= max(4, len(segments) // 4):
        segments, segment_files = zip(*active_pairs)
        segments = list(segments)
        segment_files = list(segment_files)
    emit("preprocess", 1.0, f"Extracted {len(segments)} segments ({len(segments) * 4:.0f}s total)")

    # ── Extract mel + F0 features ─────────────────────────────────
    emit("extract_f0", 0.0, "Extracting pitch and spectral features…")

    def extract_feat(seg):
        try:
            # fmax=8000 captures the full soprano/tenor range and breathiness harmonics
            f0, _, _ = librosa.pyin(seg, fmin=60, fmax=8000, sr=PROC_SR)
            f0v = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            f0s = np.array([f0v.mean() if len(f0v) else 0.0,
                            f0v.std()  if len(f0v) else 0.0,
                            float(np.percentile(f0v, 25)) if len(f0v) >= 4 else 0.0,
                            float(np.percentile(f0v, 75)) if len(f0v) >= 4 else 0.0],
                           dtype=np.float32)
        except Exception:
            f0s = np.zeros(4, dtype=np.float32)
        # 80 mel bins at 22 kHz + n_fft=1024 → finer frequency detail for timbre/emotion
        mel  = librosa.feature.melspectrogram(y=seg, sr=PROC_SR, n_mels=80, n_fft=1024, hop_length=256)
        mdb  = librosa.power_to_db(mel, ref=np.max)
        mfcc = librosa.feature.mfcc(y=seg, sr=PROC_SR, n_mfcc=20, n_fft=1024, hop_length=256)
        # delta-MFCCs: capture how features change over time (emotion lives in dynamics)
        delta = librosa.feature.delta(mfcc)
        # Spectral contrast: harmonic-vs-noise ratio per band — key for "naturalness"
        try:
            sc = librosa.feature.spectral_contrast(y=seg, sr=PROC_SR, n_fft=1024, hop_length=256)
        except Exception:
            sc = np.zeros((7, 1), dtype=np.float32)
        # Spectral rolloff: captures voice brightness (mean + std)
        try:
            ro = librosa.feature.spectral_rolloff(y=seg, sr=PROC_SR, hop_length=256)
        except Exception:
            ro = np.zeros((1, 1), dtype=np.float32)
        return np.concatenate([mdb.mean(1), mdb.std(1),
                                mfcc.mean(1), mfcc.std(1),
                                delta.mean(1),
                                sc.mean(1),  sc.std(1),
                                ro.mean(1),  ro.std(1),
                                f0s]).astype(np.float32)

    feats = []
    for i, seg in enumerate(segments):
        if i % max(1, len(segments) // 20) == 0:
            emit("extract_f0", (i + 1) / len(segments), f"Features {i + 1}/{len(segments)}")
        feats.append(extract_feat(seg))

    X   = np.array(feats, dtype=np.float32)
    mu  = X.mean(0)
    sig = X.std(0) + 1e-8
    Xn  = (X - mu) / sig
    emit("extract_f0", 1.0, f"Features shape: {X.shape[0]} × {X.shape[1]}d")

    # ── Feature normalization (prep for training) ─────────────────
    emit("extract_feats", 0.5, "Features normalized — ready for training")

    # ── Train speaker encoder ─────────────────────────────────────
    emit("train", 0.0, f"Training speaker encoder ({epochs} epochs)…")
    losses_log = []
    all_embs_np = None
    best_idxs = list(range(min(5, len(segments))))

    if HAS_TORCH and len(Xn) >= 4:
        D = Xn.shape[1]
        E = min(256, D)

        class SpeakerEnc(nn.Module):
            """Residual speaker encoder — more capacity for complex voice characteristics."""
            def __init__(self):
                super().__init__()
                self.proj = nn.Sequential(
                    nn.Linear(D, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1)
                )
                # Residual block: lets the network deepen without gradient vanishing
                self.res = nn.Sequential(
                    nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
                    nn.Linear(512, 512), nn.LayerNorm(512),
                )
                self.out = nn.Sequential(
                    nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
                    nn.Linear(256, E),
                )
            def forward(self, x):
                h = self.proj(x)
                h = nn.functional.gelu(h + self.res(h))
                return nn.functional.normalize(self.out(h), dim=-1)

        enc      = SpeakerEnc()
        # Learnable log-temperature (CLIP-style): optimises the NCE sensitivity
        log_temp = nn.Parameter(torch.tensor(float(np.log(0.1))))
        opt      = optim.AdamW(list(enc.parameters()) + [log_temp], lr=1e-3, weight_decay=1e-4)
        WARMUP   = max(5, epochs // 15)
        sch      = optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[
                optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=WARMUP),
                optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - WARMUP), eta_min=1e-5),
            ],
            milestones=[WARMUP],
        )
        Xt = torch.from_numpy(Xn)

        def nce_paired(emb, temp):
            """Symmetric InfoNCE for (original, augmented) view pairs.

            emb[:half] = original views, emb[half:] = augmented views.
            For each i: original[i] should be most similar to augmented[i].
            Cross-similarity matrix has no diagonal-masking issue because we
            compare two separate sets rather than a set against itself.
            """
            half = emb.shape[0] // 2
            if half < 2:
                return torch.zeros(1, requires_grad=True)
            e1  = emb[:half]            # original views
            e2  = emb[half:]            # augmented views
            sim = torch.mm(e1, e2.t()) / temp   # (half × half)
            lbl = torch.arange(half)
            # Symmetric: loss in both directions, averaged
            return (nn.functional.cross_entropy(sim,   lbl) +
                    nn.functional.cross_entropy(sim.t(), lbl)) * 0.5

        for ep in range(epochs):
            enc.train()
            idx    = torch.randperm(len(Xt))[:min(batch_size, len(Xt))]
            b      = Xt[idx]
            # Two augmented views: light noise + scale jitter (simulates pitch/speed shift)
            noise  = torch.randn_like(b) * 0.005
            jitter = 1.0 + (torch.rand(1).item() - 0.5) * 0.04
            views  = torch.cat([b, b * jitter + noise])  # (2*batch, D)
            temp   = torch.exp(log_temp).clamp(0.02, 0.3)
            opt.zero_grad()
            lv     = nce_paired(enc(views), temp)
            lv.backward()
            nn.utils.clip_grad_norm_(list(enc.parameters()) + [log_temp], 1.0)
            opt.step()
            sch.step()
            losses_log.append(float(lv))
            if ep % max(1, epochs // 25) == 0 or ep == epochs - 1:
                emit("train", (ep + 1) / epochs,
                     f"Epoch {ep + 1}/{epochs} — loss {lv:.4f} τ={float(temp):.3f}",
                     loss=float(lv))

        enc.eval()
        with torch.no_grad():
            all_embs_t = enc(Xt)
        all_embs_np = all_embs_t.numpy()
        spk_emb     = all_embs_np.mean(0)
        dists       = np.linalg.norm(all_embs_np - spk_emb, axis=1)
        best_idxs   = np.argsort(dists)[:7].tolist()

        torch.save({
            "type":          "maz_speaker_v2",
            "model_state":   enc.state_dict(),
            "feat_mean":     torch.from_numpy(mu),
            "feat_std":      torch.from_numpy(sig),
            "speaker_emb":   torch.from_numpy(spk_emb),
            "best_seg_idxs": best_idxs,
            "n_segments":    len(segments),
            "losses":        losses_log[-100:],
        }, str(output_dir / f"{model_name}.pth"))
        emit("train", 1.0, f"Training done — final loss {losses_log[-1]:.4f}", loss=losses_log[-1])

    else:
        emit("train", 0.5, "Computing speaker embedding (lightweight mode)…")
        spk_emb   = Xn.mean(0)
        dists     = np.linalg.norm(Xn - spk_emb, axis=1)
        best_idxs = np.argsort(dists)[:5].tolist()
        try:
            import torch as _t
            _t.save({"type": "maz_speaker_v1_lite",
                     "speaker_emb": spk_emb.tolist(),
                     "best_seg_idxs": best_idxs, "n_segments": len(segments)},
                    str(output_dir / f"{model_name}.pth"))
        except Exception:
            pass
        emit("train", 1.0, "Speaker embedding computed")

    # ── Finalize: build reference WAV from original-quality separated vocals ─────
    # We load the best source files at their NATIVE sample rate (typically 44100 Hz
    # from BSRoformer), avoiding the PROC_SR→target_sr double-resampling chain.
    # target_sr should be 44100 to match Seed-VC's internal rate with f0-condition=True.
    # Seed-VC hard-caps the reference at 25 s, so we target ≤ 24 s to be safe.
    emit("finalize", 0.0, "Creating voice reference audio…")
    try:
        def _trim_silence(y, thr=0.008):
            mask = np.abs(y) > thr
            if not mask.any():
                return y
            return y[np.argmax(mask) : len(mask) - np.argmax(mask[::-1])]

        def _xfade(a, b, fade):
            fade = min(fade, len(a), len(b))
            if fade <= 0:
                return np.concatenate([a, b])
            fo = np.linspace(1, 0, fade, dtype=np.float32)
            fi = np.linspace(0, 1, fade, dtype=np.float32)
            return np.concatenate([a[:-fade], a[-fade:] * fo + b[:fade] * fi, b[fade:]])

        # Collect unique source files in best-first order
        MAX_REF_S = 24   # seconds — stay under Seed-VC's 25 s hard cap
        seen_files, best_src_files = set(), []
        for idx in best_idxs:
            f = segment_files[idx]
            if f not in seen_files:
                seen_files.add(f)
                best_src_files.append(f)

        ref_parts, ref_native_sr = [], None
        for fpath in best_src_files:
            try:
                y_hi, sr_hi = sf.read(str(fpath))
                if y_hi.ndim > 1:
                    y_hi = y_hi.mean(1).astype(np.float32)
                if ref_native_sr is None:
                    ref_native_sr = sr_hi
                elif sr_hi != ref_native_sr:
                    y_hi = librosa.resample(y_hi, orig_sr=sr_hi, target_sr=ref_native_sr)
                y_hi = _trim_silence(y_hi.astype(np.float32))
                pk = float(np.max(np.abs(y_hi)))
                if pk > 0:
                    y_hi = y_hi / pk * 0.92
                ref_parts.append(y_hi)
            except Exception:
                continue

        if ref_parts and ref_native_sr:
            FADE_HI = int(0.03 * ref_native_sr)
            combined = ref_parts[0]
            for part in ref_parts[1:]:
                if len(combined) / ref_native_sr >= MAX_REF_S:
                    break
                combined = _xfade(combined, part, FADE_HI)
            # Hard-trim to MAX_REF_S
            combined = combined[:int(MAX_REF_S * ref_native_sr)]
            try:
                import noisereduce as nr
                combined = nr.reduce_noise(y=combined, sr=ref_native_sr, stationary=True,
                                           prop_decrease=0.5, n_fft=2048, n_jobs=1)
            except Exception:
                pass
            pk = float(np.max(np.abs(combined)))
            if pk > 0:
                combined = combined / pk * 0.92
            out_sr = target_sr if target_sr != ref_native_sr else ref_native_sr
            if out_sr != ref_native_sr:
                combined = librosa.resample(combined, orig_sr=ref_native_sr, target_sr=out_sr)
            ref_path = output_dir / f"{model_name}_ref.wav"
            sf.write(str(ref_path), combined, out_sr)
            emit("finalize", 0.7, f"Reference audio saved ({len(combined)/out_sr:.1f}s at {out_sr}Hz, native {ref_native_sr}Hz)")
        else:
            raise RuntimeError("No source files loaded for reference — falling back")

    except Exception as e:
        emit("finalize", 0.4, f"Hi-quality reference failed ({e}), falling back to segment blend…")
        try:
            n_blend = min(7, len(best_idxs))
            combined = np.concatenate([segments[i] for i in best_idxs[:n_blend]])
            combined = combined[:int(24 * PROC_SR)]  # cap at 24 s
            try:
                import noisereduce as nr
                combined = nr.reduce_noise(y=combined, sr=PROC_SR, stationary=True,
                                           prop_decrease=0.5, n_fft=2048, n_jobs=1)
            except Exception:
                pass
            pk = float(np.max(np.abs(combined)))
            if pk > 0:
                combined = combined / pk * 0.92
            if target_sr != PROC_SR:
                combined = librosa.resample(combined, orig_sr=PROC_SR, target_sr=target_sr)
            ref_path = output_dir / f"{model_name}_ref.wav"
            sf.write(str(ref_path), combined, target_sr)
            emit("finalize", 0.7, f"Reference audio saved (fallback, {len(combined)/target_sr:.1f}s at {target_sr}Hz)")
        except Exception as e2:
            emit("finalize", 0.7, f"Reference audio warning: {e2}")

    # ── FAISS index ───────────────────────────────────────────────
    try:
        import faiss
        embs_i = (all_embs_np if all_embs_np is not None else Xn).astype(np.float32)
        fidx   = faiss.IndexFlatL2(embs_i.shape[1])
        fidx.add(embs_i)
        faiss.write_index(fidx, str(output_dir / f"{model_name}.index"))
        emit("finalize", 0.95, f"Speaker index: {fidx.ntotal} vectors (dim={embs_i.shape[1]})")
    except ImportError:
        emit("finalize", 0.95, "faiss-cpu not installed — index skipped (pip install faiss-cpu)")
    except Exception as e:
        emit("finalize", 0.95, f"Index warning: {e}")

    emit("done", 1.0, f"Training complete — model: {model_name}")


if __name__ == "__main__":
    main()
