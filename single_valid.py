
import argparse
import torch
import numpy as np
from dataset import CombinedSketchDataset, preprocess_sequence
from model   import SketchLSTM

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load validationset
    ds = CombinedSketchDataset("data/subsampledv1", split="valid", pad_length=150)
    
    idx = max(0, min(args.index, len(ds)-1))
    x_seq, true_label = ds[idx]

  
    if isinstance(x_seq, np.ndarray):
        tensor = torch.from_numpy(x_seq.astype(np.float32))
    elif isinstance(x_seq, torch.Tensor):
        tensor = x_seq.float()
    else:
        tensor = torch.tensor(x_seq, dtype=torch.float32)

    x = tensor.unsqueeze(0).to(device) 

    # load model
    model = SketchLSTM(input_dim=3, hidden_dim=512, num_layers=2, dropout=0.5,
                       num_classes=len(ds.files))
    state = torch.load("models/final_incremental.pth", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # inference
    with torch.no_grad():
        logits = model(x)[0]
        probs  = torch.softmax(logits, dim=0).cpu().numpy()

    pred_idx = int(probs.argmax())
    print(f"Sample #{idx}:")
    print(f"  True label:      {ds.files[true_label]}")
    print(f"  Predicted label: {ds.files[pred_idx]}")
    print(f"  Confidence:      {probs[pred_idx]*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single validation sample")
    parser.add_argument(
        "--index", "-i",
        type=int,
        default=0,
        help="Validation sample index (default: 0)"
    )
    args = parser.parse_args()
    main(args)
