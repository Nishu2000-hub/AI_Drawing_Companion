#!/usr/bin/env python
"""
Run a single test sample through the trained model and print results.

Usage:
  python single_test.py [--index INDEX]

If --index is omitted, defaults to 0 (the first test sample).
"""
import argparse
import torch
import numpy as np
from dataset import CombinedSketchDataset, preprocess_sequence
from model   import SketchLSTM

def main(args):
    # select device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load the test dataset (split="test")
    ds = CombinedSketchDataset("data/subsampledv1", split="test", pad_length=150)

    # clamp index into valid range
    idx = max(0, min(args.index, len(ds)-1))
    seq, true_label = ds[idx]

    # ensure float32 tensor
    if isinstance(seq, np.ndarray):
        tensor = torch.from_numpy(seq.astype(np.float32))
    else:
        tensor = seq.float() if isinstance(seq, torch.Tensor) else torch.tensor(seq, dtype=torch.float32)

    x = tensor.unsqueeze(0).to(device)  # add batch dimension

    # load the final incremental model
    model = SketchLSTM(input_dim=3, hidden_dim=512, num_layers=2, dropout=0.5,
                       num_classes=len(ds.files))
    state = torch.load("models/final_incremental.pth", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # run inference
    with torch.no_grad():
        logits = model(x)[0]
        probs  = torch.softmax(logits, dim=0).cpu().numpy()

    pred_idx = int(probs.argmax())
    print(f"Test Sample #{idx}:")
    print(f"  True label:      {ds.files[true_label]}")
    print(f"  Predicted label: {ds.files[pred_idx]}")
    print(f"  Confidence:      {probs[pred_idx]*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single test sample")
    parser.add_argument(
        "--index", "-i",
        type=int,
        default=0,
        help="Test sample index (default: 0)"
    )
    args = parser.parse_args()
    main(args)
