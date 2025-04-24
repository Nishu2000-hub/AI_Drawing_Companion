import os

raw_dir = r'data/slim'
files = sorted(f for f in os.listdir(raw_dir) if f.endswith('.npz'))
print(f"Total .npz files found: {len(files)}")
print("First 5 categories:", files[:5])
