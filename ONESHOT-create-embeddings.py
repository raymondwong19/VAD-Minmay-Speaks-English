#!/usr/bin/env python3
"""
ONESHOT-create-embeddings.py

Compute speaker embeddings with an ONNX model that expects features [B, T, 80].
Accepts only .wav files. Produces an NPZ containing: average (256,), names (N,), embeddings (N,256).

Usage:
  ONESHOT-create-embeddings.py /path/to/enroll-dir
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import onnxruntime as ort
from scipy.signal import resample_poly
import math

# CONFIG
ONNX_PATH = "pyannote/cnceleb_resnet34-wespeaker.onnx"
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 512
HOP_MS = 10
WIN_MS = 25
CACHE_NAME = "enroll_embeddings.npz"
AVERAGE_ACROSS_FILES = True
FORCE_CPU = True
SUPPORTED_EXTS = {".wav"}  # only WAV now
EPS = 1e-6


def load_wav(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    wav, orig_sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if orig_sr != sr:
        wav = resample_audio_poly(wav, orig_sr, sr)
    return wav.astype(np.float32)


def resample_audio_poly(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    up = target_sr
    down = orig_sr
    g = np.gcd(up, down)
    up //= g
    down //= g
    y_rs = resample_poly(y, up, down)
    return y_rs.astype(np.float32)


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
    S = np.abs(fft) ** 2  # power
    mel_basis = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_spec = (mel_basis @ S.T).T
    log_mel = np.log(np.maximum(mel_spec, EPS))
    return log_mel.astype(np.float32)


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


def normalize_feats(feats: np.ndarray) -> np.ndarray:
    mu = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return ((feats - mu) / std).astype(np.float32)


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
        emb_batch = out0          # (B, D)
    elif out0.ndim == 3:
        emb_batch = out0.mean(axis=1)  # (B, D)
    else:
        raise RuntimeError(f"Unexpected model output shape: {out0.shape}")
    emb = emb_batch[0].astype(np.float32)
    norm = np.linalg.norm(emb, keepdims=True)
    if norm == 0:
        norm = 1.0
    emb = emb / norm
    return emb


def main(enroll_dir: str) -> int:
    enroll_path = Path(enroll_dir)
    if not enroll_path.is_dir():
        print("Enroll directory not found:", enroll_path)
        return 2
    model_path = Path(ONNX_PATH)
    if not model_path.is_file():
        print("ONNX model not found at:", model_path)
        return 3
    cache_path = enroll_path / CACHE_NAME
    if cache_path.exists():
        print("Cache already exists at:", cache_path)
        return 0
    sess = create_session(model_path, force_cpu=FORCE_CPU)
    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    print("ONNX input:", input_meta.name, input_meta.shape, input_meta.type)
    print("Using providers:", sess.get_providers())
    files = sorted([p for p in enroll_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])
    if not files:
        print("No WAV audio files found in", enroll_path)
        return 4
    names = []
    embs = []
    for p in files:
        try:
            wav = load_wav(str(p))
        except Exception as e:
            print(f"Failed to load {p.name}: {e}")
            continue
        try:
            emb = infer_emb_from_wav(sess, input_name, wav)
        except Exception as e:
            print(f"Failed to run ONNX on {p.name}: {e}")
            continue
        names.append(p.name)
        embs.append(emb)
        print(f"Processed {p.name} -> {emb.shape}")
    if not embs:
        print("No embeddings produced.")
        return 5
    embs_arr = np.stack(embs, axis=0)  # (N, 256)
    if AVERAGE_ACROSS_FILES:
        avg = embs_arr.mean(axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-12)
        np.savez_compressed(cache_path, average=avg, names=np.array(names), embeddings=embs_arr)
        print("Saved average embedding and per-file embeddings to:", cache_path)
    else:
        np.savez_compressed(cache_path, names=np.array(names), embeddings=embs_arr)
        print("Saved per-file embeddings to:", cache_path)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ONESHOT-create-embeddings.py /path/to/enroll-dir")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
