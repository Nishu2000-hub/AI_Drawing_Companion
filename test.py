from dataset import CombinedSketchDataset

ds = CombinedSketchDataset(
    data_dir="data/subsampledv1",
    split="train",
    pad_length=100
)

print("Total train samples:", len(ds))
x, y = ds[0]
print("Sample shape:", x.shape, "Label:", y)
