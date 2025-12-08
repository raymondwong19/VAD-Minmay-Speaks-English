#!/usr/bin/env python3
"""
voiceprint_part2.py

Compute an average embedding for all .wav files in a target directory using the same ONNX model
and preprocessing as the enroll script, then compare (cosine similarity) it to the enroll average.

Usage:
  python3 voiceprint_part2.py /path/to/target-dir

Output (stdout): CSV with header:
filename_count,dir_average_score,top_file_score,top_file_name

Notes:
- Expects enroll cache at: data/enroll-en/enroll_embeddings.npz (contains 'average' and optional 'embeddings'/'names').
- Accepts only .wav files.
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import onnxruntime as ort
from scipy.signal import resample_poly
import math

# CONFIG (match your patched embed cache script)
ONNX_PATH = "pyannote/cnceleb_resnet34-wespeaker.onnx"
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 512
HOP_MS = 10
WIN_MS = 25
EPS = 1e-6
FORCE_CPU = True
SUPPORTED_EXTS = {".wav"}

# --- audio / feature helpers (same as in patched script) ---
def resample_audio_poly(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    up = target_sr
    down = orig_sr
    g = np.gcd(up, down)
    up //= g
    down //= g
    y_rs = resample_poly(y, up, down)
    return y_rs.astype(np.float32)

def load_wav(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    wav, orig_sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if orig_sr != sr:
        wav = resample_audio_poly(wav, orig_sr, sr)
    return wav.astype(np.float32)

def mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel):
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)
    fmin = 0.0
    fmax = sr / 2.0
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left = bins[i - 1]
        center = bins[i]
        right = bins[i + 1]
        if center > left:
            fb[i - 1, left:center] = (np.arange(left, center) - left) / max(1, center - left)
        if right > center:
            fb[i - 1, center:right] = (right - np.arange(center, right)) / max(1, right - center)
    return fb

def wav_to_log_mel(y: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = N_MELS,
                   n_fft: int = N_FFT, win_ms: int = WIN_MS, hop_ms: int = HOP_MS) -> np.ndarray:
    hop_length = int(sr * hop_ms / 1000)
    win_length = int(sr * win_ms / 1000)
    if len(y) < win_length:
        pad = win_length - len(y)
        y = np.concatenate([y, np.zeros(pad, dtype=y.dtype)])
    num_frames = 1 + int(math.ceil(max(0, (len(y) - win_length) / hop_length)))
    desired_len = (num_frames - 1) * hop_length + win_length
    if desired_len > len(y):
        pad = desired_len - len(y)
        y = np.concatenate([y, np.zeros(pad, dtype=y.dtype)])
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(num_frames, win_length),
        strides=(y.strides[0] * hop_length, y.strides[0])
    ).copy()
    window = np.hanning(win_length).astype(y.dtype)
    frames *= window[None, :]
    fft = np.fft.rfft(frames, n=n_fft, axis=1)
    S = np.abs(fft) ** 2
    mel_basis = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_spec = (mel_basis @ S.T).T
    log_mel = np.log(np.maximum(mel_spec, EPS))
    return log_mel.astype(np.float32)

def normalize_feats(feats: np.ndarray) -> np.ndarray:
    mu = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return ((feats - mu) / std).astype(np.float32)

# --- ONNX session & inference ---
def create_session(path: Path, force_cpu: bool = True) -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"] if force_cpu else ort.get_available_providers()
    sess = ort.InferenceSession(str(path), providers=providers)
    return sess

def infer_emb_from_wav(sess: ort.InferenceSession, input_name: str, wav: np.ndarray) -> np.ndarray:
    feats = wav_to_log_mel(wav)
    if feats.shape[1] != N_MELS:
        raise RuntimeError(f"Unexpected feature dim: {feats.shape[1]}, expected {N_MELS}")
    feats = normalize_feats(feats)
    inp = feats[None, :, :]  # (1, T, 80)
    out = sess.run(None, {input_name: inp})
    out0 = out[0]
    if out0.ndim == 2:
        emb_batch = out0
    elif out0.ndim == 3:
        emb_batch = out0.mean(axis=1)
    else:
        raise RuntimeError(f"Unexpected model output shape: {out0.shape}")
    emb = emb_batch[0].astype(np.float32)
    n = np.linalg.norm(emb)
    if n == 0:
        return emb
    return emb / n

# --- utilities ---
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# --- main ---
def main(target_dir: str) -> int:
    target = Path(target_dir)
    if not target.is_dir():
        print("Target directory not found:", target)
        return 2
    enroll_cache = Path("data/enroll-en/enroll_embeddings.npz")
    if not enroll_cache.is_file():
        print("Enroll cache not found:", enroll_cache)
        return 3
    data = np.load(enroll_cache, allow_pickle=True)
    if "average" not in data:
        print("Enroll cache missing 'average' vector.")
        return 4
    enroll_avg = data["average"].astype(np.float32)
    # ensure normalized
    enroll_avg = enroll_avg / (np.linalg.norm(enroll_avg) + 1e-12)

    sess = create_session(Path(ONNX_PATH), force_cpu=FORCE_CPU)
    input_name = sess.get_inputs()[0].name

    wavs = sorted([p for p in target.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])
    if not wavs:
        print("No WAV files found in target dir:", target)
        return 5

    per_file_embs = []
    per_file_names = []
    for p in wavs:
        try:
            wav = load_wav(str(p))
            emb = infer_emb_from_wav(sess, input_name, wav)
            per_file_embs.append(emb)
            per_file_names.append(p.name)
        except Exception as e:
            print(f"ERR loading/infer {p.name}: {e}", file=sys.stderr)

    if not per_file_embs:
        print("No embeddings produced for target files.", file=sys.stderr)
        return 6

    embs_arr = np.stack(per_file_embs, axis=0)  # (N, D)
    # directory average
    dir_avg = embs_arr.mean(axis=0)
    dir_avg = dir_avg / (np.linalg.norm(dir_avg) + 1e-12)
    # compare to enroll average
    score = cosine(dir_avg, enroll_avg)
    # also report best per-file match score (optional but helpful)
    per_scores = (embs_arr @ enroll_avg)
    top_idx = int(np.argmax(per_scores))
    top_score = float(per_scores[top_idx])
    top_name = per_file_names[top_idx]

    # print CSV header then single line
    print("file_count,dir_average_score,top_file_score,top_file_name")
    print(f"{len(per_file_names)},{score:.6f},{top_score:.6f},{top_name}")

    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 voiceprint_part2.py /path/to/target-dir")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
