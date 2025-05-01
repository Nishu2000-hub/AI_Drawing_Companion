import os

for v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[v] = "4"

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader

from dataset import CombinedSketchDataset, preprocess_sequence
from model   import SketchLSTM


if torch.cuda.is_available():
    torch.cuda.set_device(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)


DATA_DIR      = r"data\subsampledv1"
CHECKPOINT    = r"models\final_incremental.pth"
PAD_LENGTH    = 150
UNKNOWN_DIR   = r"data\unknown"
CONF_THRESHOLD   = 0.40
ENERGY_THRESHOLD = 6.681478602899007
BATCH_SIZE    = 128
PLOTS_DIR     = r"plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


class_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.npz'))
num_classes = len(class_files)
model = SketchLSTM(input_dim=3, hidden_dim=512, num_layers=2,
                   dropout=0.5, num_classes=num_classes)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE).eval()

def energy_score(logits: torch.Tensor) -> np.ndarray:
    return -torch.logsumexp(logits, dim=1).cpu().numpy()


test_ds = CombinedSketchDataset(DATA_DIR, split="test", pad_length=PAD_LENGTH)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0, pin_memory=False)

all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE).float()
        logits = model(x)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"Closed-set Test Accuracy: {acc:.4f}")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,8))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'))
plt.close()


known_energies, known_conf = [], []
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(DEVICE).float()
        logits = model(x)
        known_energies.extend(energy_score(logits))
        known_conf.extend(torch.softmax(logits,1).max(1)[0].cpu().numpy())

unk_energies, unk_conf = [], []
for fn in sorted(os.listdir(UNKNOWN_DIR)):
    seq = np.load(os.path.join(UNKNOWN_DIR, fn))
    proc = preprocess_sequence(seq, pad_length=PAD_LENGTH)
    x = torch.from_numpy(proc).unsqueeze(0).to(DEVICE).float()
    with torch.no_grad():
        logits = model(x)
        unk_energies.extend(energy_score(logits))
        unk_conf.extend(torch.softmax(logits,1).max(1)[0].cpu().numpy())


y_true  = np.concatenate([np.ones(len(known_energies)), np.zeros(len(unk_energies))])
y_scores = np.concatenate([
    -(np.array(known_energies) - ENERGY_THRESHOLD),
    -(np.array(unk_energies)   - ENERGY_THRESHOLD)
])
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Novelty Detection ROC')
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, 'novelty_roc.png'))
plt.close()

print("Evaluation complete. Plots saved to 'plots/'")
