#!/usr/bin/env python3
"""
NN-Shape-Tester.py

Quick tool to probe an ONNX model's input shape and to try feeding
a few different shaped random tensors to see which shapes the runtime accepts.

Usage:
  python NN-Shape-Tester.py model.onnx
"""
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

TEST_SHAPES = [
    # common candidates: (B, T, 80), (1, T, 80), (B, 80, T), (1, 80, T), (B, 1, T, 80)
    (1, 100, 80),
    (1, 200, 80),
    (4, 100, 80),
    (1, 80, 100),
    (1, 1, 100, 80),
    (1, 80),
]

def try_run(sess, input_name, shape, dtype=np.float32):
    x = np.random.randn(*shape).astype(dtype)
    try:
        out = sess.run(None, {input_name: x})
        return True, f"OK -> outputs: {len(out)}; shapes: {[o.shape for o in out]}"
    except Exception as e:
        return False, str(e)

def main(path):
    p = Path(path)
    if not p.is_file():
        print("Model not found:", p)
        return 2

    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    print("Model inputs:")
    for i, inp in enumerate(inputs):
        print(f" {i}: name='{inp.name}' shape={inp.shape} type={inp.type}")
    print("Model outputs:")
    for i, out in enumerate(sess.get_outputs()):
        print(f" {i}: name='{out.name}' shape={out.shape} type={out.type}")

    # If model has a single input, test various shapes by trying to map dimensions.
    if len(inputs) != 1:
        print("Note: model has multiple inputs — this tester only auto-trials single-input cases.")
        return 0

    input_meta = inputs[0]
    input_name = input_meta.name
    print("\nAttempting test shapes:")
    for s in TEST_SHAPES:
        ok, msg = try_run(sess, input_name, s)
        print(f" try shape {s}: {'SUCCESS' if ok else 'FAIL'} -> {msg}")

    # Also try a couple of variable-length T values if input shape shows 'T' or None.
    # Build shapes by inspecting metadata for None or symbolic dims
    meta_shape = tuple(input_meta.shape)
    if any(d is None or (isinstance(d, str) and d not in ('B',)) for d in meta_shape):
        # construct shapes replacing batch with 1 and symbolic/time dims with various lengths
        def normalize_dim(d, test_len):
            if d is None or isinstance(d, str):
                return test_len
            return int(d)
        for T in (50, 100, 400):
            trial = tuple(normalize_dim(d, T) for d in meta_shape)
            ok, msg = try_run(sess, input_name, trial)
            print(f" trial symbolic-> {trial}: {'SUCCESS' if ok else 'FAIL'} -> {msg}")

    print("\nDone.")
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: NN-Shape-Tester.py /path/to/model.onnx")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
