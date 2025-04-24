import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import CombinedSketchDataset
from model import SketchLSTM  # assuming model.py exports this

def train_base(
    data_dir: str = "data/subsampled",
    model_dir: str = "models",
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    num_epochs: int = 30,
    patience: int = 5,
    max_grad_norm: float = 5.0,
    pad_length: int = 100,
    device: str = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(model_dir, exist_ok=True)

    # Datasets & Loaders
    train_ds = CombinedSketchDataset(data_dir, split="train", pad_length=pad_length)
    val_ds   = CombinedSketchDataset(data_dir, split="valid", pad_length=pad_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Model, loss, optimizer, scheduler
    model = SketchLSTM(input_dim=3, hidden_dim=512, num_layers=2, dropout=0.5,
                       num_classes=len(train_ds.files)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                  factor=0.5, patience=2, verbose=True)

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # --- Training ---
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            x, y = x.to(device).float(), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            _, preds = logits.max(1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                x, y = x.to(device).float(), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                _, preds = logits.max(1)
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)

        val_loss = val_loss / val_total
        val_acc  = val_correct / val_total

        print(f"\nEpoch {epoch}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        scheduler.step(val_acc)

        # --- Checkpoint & Early Stopping ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(model_dir, "best_base.pth"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"No improvement for {patience} epochs. Stopping.")
            break

    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    train_base()
