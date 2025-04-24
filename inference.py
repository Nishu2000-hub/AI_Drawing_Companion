import os
import threading
import subprocess
import tkinter as tk
from tkinter import simpledialog, messagebox
import numpy as np
import torch
from tkinter import Scale, HORIZONTAL

from model import SketchLSTM
from dataset import preprocess_sequence

DATA_DIR = "data/subsampledv1"
CHECKPOINT = "models/final_incremental.pth"
PAD_LENGTH = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UNKNOWN_SAVE_DIR = "data/unknown"
ENSEMBLE_SIZE = 5
DEFAULT_THRESHOLD = 6.681478602899007  # precomputed μ + 2σ
CONF_THRESHOLD = 0.60  # softmax confidence floor
TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "train_ci.py")

#helper functions
def energy_score(logits: torch.Tensor) -> float:
    """Compute energy score E(x) = -log(sum exp(logits))"""
    return -torch.logsumexp(logits, dim=0).item()


def smooth_sequence(seq: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply moving-average smoothing to dx, dy channels"""
    seq_sm = seq.copy()
    for axis in (0,1):
        seq_sm[:, axis] = np.convolve(seq[:, axis], np.ones(window)/window, mode='same')
    return seq_sm

# model  Loading 
class_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.npz'))
class_names = [os.path.splitext(f)[0] for f in class_files]
num_classes = len(class_names)

# Initialize and load model
model = SketchLSTM(input_dim=3, hidden_dim=512, num_layers=2,
                   dropout=0.5, num_classes=num_classes)

def reload_model():
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

reload_model()

# Ensure unknown save directory exists
os.makedirs(UNKNOWN_SAVE_DIR, exist_ok=True)


def run_incremental_training():
    try:
        subprocess.run(["python", TRAIN_SCRIPT], check=True)
        reload_model()
        messagebox.showinfo("Retrained", "Model has been updated with new class.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Training Error", f"Incremental training failed:\n{e}")

# GUI 
class SketchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interactive Sketch Recognizer")

        # Drawing canvas
        self.canvas = tk.Canvas(self, width=400, height=400, bg='white')
        self.canvas.pack(padx=5, pady=5)

        # Control buttons and threshold slider
        ctrl = tk.Frame(self)
        ctrl.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(ctrl, text="Classify", command=self.classify).pack(side=tk.LEFT, expand=True)
        tk.Button(ctrl, text="Clear",    command=self.clear).pack(side=tk.LEFT, expand=True)
        self.threshold_slider = Scale(
            ctrl, from_=0.0, to=DEFAULT_THRESHOLD*2, resolution=0.1,
            orient=HORIZONTAL, label="Energy Threshold"
        )
        self.threshold_slider.set(DEFAULT_THRESHOLD)
        self.threshold_slider.pack(fill=tk.X, padx=5)

        # State for strokes
        self.paths = []
        self.current = []
        self.canvas.bind('<ButtonPress-1>',    self.on_press)
        self.canvas.bind('<B1-Motion>',        self.on_move)
        self.canvas.bind('<ButtonRelease-1>',  self.on_release)

    def on_press(self, event):
        self.current = [(event.x, event.y)]

    def on_move(self, event):
        x0, y0 = self.current[-1]
        self.current.append((event.x, event.y))
        self.canvas.create_line(x0, y0, event.x, event.y)

    def on_release(self, event):
        self.current.append((event.x, event.y))
        self.paths.append(list(self.current))

    def clear(self):
        self.canvas.delete('all')
        self.paths = []

    def classify(self):
        if not self.paths:
            messagebox.showinfo("Info", "Draw something first.")
            return

        # Convert strokes to Δx,Δy,pen sequence
        seq = []
        px, py = self.paths[0][0]
        for stroke in self.paths:
            for x, y in stroke:
                seq.append([x-px, y-py, 0])
                px, py = x, y
            seq.append([0, 0, 1])
        seq = np.array(seq, dtype=np.float32)

        # Smooth and ensemble jittered predictions
        seq = smooth_sequence(seq)
        ensemble_probs = torch.zeros(num_classes, device=DEVICE)
        for _ in range(ENSEMBLE_SIZE):
            jitter = seq.copy()
            jitter[:, :2] += np.random.randn(*jitter[:, :2].shape) * 0.3
            proc = preprocess_sequence(jitter, pad_length=PAD_LENGTH)
            x = torch.from_numpy(proc).unsqueeze(0).to(DEVICE).float()
            with torch.no_grad():
                logits = model(x)[0]
                ensemble_probs += torch.softmax(logits, dim=0)
        probs = (ensemble_probs / ENSEMBLE_SIZE).cpu()

        # Compute energy score and max confidence
        energy = energy_score(torch.log(probs).to(DEVICE))
        max_conf = float(probs.max().item())

        # Prepare prediction text
        topk = torch.topk(probs, k=min(5, num_classes))
        pred_text = "\n".join(
            f"{class_names[i]}: {v.item()*100:.1f}%" 
            for i, v in zip(topk.indices.tolist(), topk.values)
        )

        threshold = self.threshold_slider.get()
        # Hybrid novelty check: energy or low confidence
        if energy > threshold or max_conf < CONF_THRESHOLD:
            if messagebox.askyesno(
                "Unknown Sketch",
                f"Energy {energy:.2f} > {threshold:.1f} or confidence {max_conf*100:.1f}% < {CONF_THRESHOLD*100:.0f}%\nTreat as new class?"
            ):
                new_label = simpledialog.askstring("Label", "Enter new class name:")
                if new_label:
                    idx = len(os.listdir(UNKNOWN_SAVE_DIR))
                    np.save(
                        os.path.join(UNKNOWN_SAVE_DIR, f"{new_label}_{idx}.npy"),
                        seq
                    )
                    messagebox.showinfo("Saved", f"Saved '{new_label}'. Starting retraining...")
                    threading.Thread(
                        target=run_incremental_training,
                        daemon=True
                    ).start()
                else:
                    messagebox.showwarning("Skipped", "No label entered.")
        else:
            messagebox.showinfo("Top Predictions", pred_text)

if __name__ == '__main__':
    app = SketchApp()
    app.mainloop()
