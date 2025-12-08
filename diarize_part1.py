#!/usr/bin/env python3
"""
Local speaker diarization (Part 1 for Minmay) using a locally saved
pyannote/speaker-diarization-community-1 model.

Precondition: you have saved the model directory locally at:
  ./pyannote/speaker-diarization-community-1
(or change LOCAL_MODEL_DIR below to your path).
I cannot redistribute the model, so you better find it yourself.
Also create your own venv in minmay-chroot.

Input must be a WAV file (16-bit, 44100KHz).

Usage:
  python diarize_part1.py /path/to/audio.wav [num_speakers] [out.json]

Requires:
  pip install pyannote.audio==4.0.2 torchaudio==2.7.0
  You'll also need torchcodec==0.7.0 (0.8.0). No Pip prebuilt for aarch64 through.
"""
import sys
import os
import json
import torchaudio
import torch
from pyannote.audio import Pipeline

# Path to local model directory
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "pyannote", "speaker-diarization-community-1")

def _enable_unpickle_for_pyannote():
    """
    Try to allowlist common pyannote classes used in checkpoints.
    If add_safe_globals is not available or we cannot import classes,
    return a (real_torch_load, patched_load) pair to monkeypatch torch.load.
    """
    try:
        # import a broad set of classes that pyannote checkpoints commonly reference
        from pyannote.audio.core.task import Specifications, Problem, Resolution, Protocol, Task
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([Specifications, Problem, Resolution, Protocol, Task])
            return None
    except Exception:
        pass

    # Fallback: monkeypatch torch.load to force weights_only=False
    real_torch_load = torch.load

    def _torch_load_allow(path, *args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return real_torch_load(path, *args, **kwargs)

    return real_torch_load, _torch_load_allow

def load_pipeline():
    if not os.path.isdir(LOCAL_MODEL_DIR):
        raise FileNotFoundError(f"Local model dir not found: {LOCAL_MODEL_DIR}")

    # Prevent HF hub network calls
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    device = torch.device("cpu")

    # Attempt to enable safe unpickling; fallback to monkeypatch if needed
    fallback = _enable_unpickle_for_pyannote()
    patched = False
    if fallback is not None:
        real_torch_load, patched_load = fallback
        torch.load = patched_load
        patched = True

    try:
        pipeline = Pipeline.from_pretrained(os.path.abspath(LOCAL_MODEL_DIR))
    finally:
        if patched:
            torch.load = real_torch_load

    return pipeline.to(device)

def diarize(pipeline, wav_path, num_speakers=0):
    waveform, sample_rate = torchaudio.load(wav_path)
    audio = {"waveform": waveform, "sample_rate": sample_rate}
    params = {"num_speakers": int(num_speakers)} if int(num_speakers) > 0 else {"min_speakers": 2, "max_speakers": 6}

    ann_out = pipeline(audio, **params)

    # Normalize: handle pyannote.audio 4.x DiarizeOutput wrappers and older Annotation returns
    annotation = None
    # Common wrapper attribute names used by newer pyannote versions
    for attr in ("speaker_diarization", "diarization", "segmentation"):
        if hasattr(ann_out, attr):
            annotation = getattr(ann_out, attr)
            break

    if annotation is None:
        if hasattr(ann_out, "to_annotation"):
            try:
                annotation = ann_out.to_annotation()
            except Exception:
                annotation = ann_out
        else:
            annotation = ann_out

    out = []

    # Try itertracks first (common signature: (segment, track, label))
    if hasattr(annotation, "itertracks"):
        for segment, _, label in annotation.itertracks(yield_label=True):
            out.append({"start": round(segment.start, 3), "end": round(segment.end, 3), "speaker_id": label})
        return out

    # Fallback to itersegments (signature often: (segment, label))
    if hasattr(annotation, "itersegments"):
        for segment, label in annotation.itersegments(yield_label=True):
            out.append({"start": round(segment.start, 3), "end": round(segment.end, 3), "speaker_id": label})
        return out

    # Last resort: try iterating items() if Annotation behaves like a dict
    try:
        for segment, label in annotation.items():
            out.append({"start": round(segment.start, 3), "end": round(segment.end, 3), "speaker_id": label})
        return out
    except Exception:
        raise RuntimeError(f"Unsupported annotation object: {type(annotation)} — cannot iterate segments")
    # Jeez so many things to implement and the behavior changed during the months

def main():
    if len(sys.argv) < 2:
        print("Usage: python diarize_part1.py <local_wav_path> [num_speakers] [out.json]")
        sys.exit(1)
    inp = sys.argv[1]
    if not os.path.exists(inp):
        print("Input file not found")
        sys.exit(1)
    num_spk = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out_json = sys.argv[3] if len(sys.argv) > 3 else "diarization.json"

    try:
        pipeline = load_pipeline()
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        sys.exit(1)

    try:
        segments = diarize(pipeline, inp, num_spk)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2)
        print(f"Wrote {len(segments)} segments to {out_json}")
    except Exception as e:
        print(f"Diarization error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
