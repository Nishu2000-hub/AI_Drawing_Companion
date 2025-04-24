import os
import numpy as np

def subsample_data(raw_dir, out_dir, num_categories=50,
                   train_n=70000, valid_n=2500, test_n=2500, seed=42):
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(raw_dir) if f.endswith('.npz'))[:num_categories]
    for fname in files:
        src_path = os.path.join(raw_dir, fname)
        # allow_pickle=True, fix_imports for Python2->3 pickles, encoding for bytes
        data = np.load(src_path, allow_pickle=True, fix_imports=True, encoding='latin1')

        out_splits = {}
        for split, n in (("train", train_n), ("valid", valid_n), ("test", test_n)):
            arr = data[split]
            if len(arr) < n:
                raise ValueError(f"{fname} has only {len(arr)} in '{split}', need {n}")
            idx = np.random.choice(len(arr), size=n, replace=False)
            out_splits[split] = arr[idx]

        dst_path = os.path.join(out_dir, fname)
        np.savez_compressed(dst_path, **out_splits)
        print(f"[OK] {fname}")

if __name__ == "__main__":
    # Use raw strings for Windows paths
    raw_dir = r"C:\Users\khand\AppData\Local\Google\Cloud SDK\sketchrnnnpz\data\slim"
    out_dir = r"C:\Users\khand\AppData\Local\Google\Cloud SDK\sketchrnnnpz\data\subsampledv1"
    subsample_data(raw_dir, out_dir)
