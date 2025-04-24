# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import os, random, copy, warnings
# for var in (
#     "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
#     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"
# ):
#     os.environ[var] = "4"

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Subset
# from torch.optim import Adam
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.cuda.amp import GradScaler, autocast
# from tqdm import tqdm

# from dataset import CombinedSketchDataset
# from model   import SketchLSTM

# warnings.filterwarnings("ignore", category=UserWarning)


# def train_incremental(
#     data_dir: str  = r"C:\Users\khand\AppData\Local\Google\Cloud SDK\sketchrnnnpz\data\subsampledv1",
#     model_dir: str = r"C:\Users\khand\AppData\Local\Google\Cloud SDK\sketchrnnnpz\models",
#     base_checkpoint: str = r"C:\Users\khand\AppData\Local\Google\Cloud SDK\sketchrnnnpz\models\best_base.pth",
#     task_size: int = 10,
#     memory_per_class: int = 20,
#     alpha: float = 1.0,
#     lr: float = 5e-4,
#     weight_decay: float = 1e-6,
#     num_epochs: int = 15,
#     patience: int = 3,
#     batch_size: int = 128,
#     pad_length: int = 100,
#     max_grad_norm: float = 5.0,
# ):
    
#     if torch.cuda.is_available():
#         torch.cuda.set_device(0)
                
#     device = torch.device("cuda:0")
#     torch.set_num_threads(4)                 
#     random.seed(42); np.random.seed(42); torch.manual_seed(42)

#     os.makedirs(model_dir, exist_ok=True)

   
#     full_train = CombinedSketchDataset(data_dir, "train", pad_length)
#     full_val   = CombinedSketchDataset(data_dir, "valid", pad_length)
#     num_classes = len(full_train.files)

#     tasks = [list(range(i, min(i+task_size, num_classes)))
#              for i in range(0, num_classes, task_size)]

   
#     memory_buffer = {}
#     first_task = tasks[0]
#     for cls in first_task:
#         idxs = [i for i,l in enumerate(full_train.labels) if l == cls]
#         memory_buffer[cls] = random.sample(idxs, memory_per_class)

    
#     model = SketchLSTM(3, 512, 2, 0.5, num_classes).to(device)
#     model.load_state_dict(torch.load(base_checkpoint, map_location=device))
#     teacher = copy.deepcopy(model).eval()

#     criterion_ce  = nn.CrossEntropyLoss()
#     criterion_mse = nn.MSELoss()
#     scaler = GradScaler()                 

   
#     for step_id, new_task in enumerate(tasks[1:], start=1):
#         print(f"\n=== Step {step_id}: classes {new_task} ===")

       
#         train_idx = [i for idxs in memory_buffer.values() for i in idxs]
#         for cls in new_task:
#             train_idx += [i for i,l in enumerate(full_train.labels) if l == cls]
#         random.shuffle(train_idx)

#         val_idx = [i for i,l in enumerate(full_val.labels)
#                    if l in (*memory_buffer.keys(), *new_task)]

#         train_loader = DataLoader(Subset(full_train, train_idx),
#                                   batch_size=batch_size, shuffle=True,
#                                   num_workers=4, pin_memory=True)
#         val_loader   = DataLoader(Subset(full_val, val_idx),
#                                   batch_size=batch_size, shuffle=False,
#                                   num_workers=4, pin_memory=True)

#         optimizer  = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#         scheduler  = ReduceLROnPlateau(optimizer, "max", 0.5, 2, verbose=True)
#         best_val   = 0.0; stale = 0

       
#         for epoch in range(1, num_epochs+1):
          
#             model.train()
#             tloss = tcorrect = ttotal = 0
#             for x,y in tqdm(train_loader, desc=f"Step{step_id}-E{epoch} train", ncols=100):
#                 x, y = x.to(device).float(), y.to(device)
#                 optimizer.zero_grad()
#                 with autocast(dtype=torch.float16):
#                     logits = model(x)
#                     loss_ce = criterion_ce(logits, y)

#                     mem_mask = torch.tensor(
#                         [lbl in memory_buffer for lbl in y.cpu()], device=device)
#                     if mem_mask.any():
#                         with torch.no_grad():
#                             old_logits = teacher(x[mem_mask])
#                         loss_dist = criterion_mse(logits[mem_mask], old_logits)
#                     else:
#                         loss_dist = torch.tensor(0., device=device)

#                     loss = loss_ce + alpha*loss_dist

#                 scaler.scale(loss).backward()
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#                 scaler.step(optimizer); scaler.update()

#                 tloss     += loss.item()*x.size(0)
#                 tcorrect  += (logits.argmax(1) == y).sum().item()
#                 ttotal    += x.size(0)

          
#             model.eval()
#             vcorrect = vtotal = 0
#             with torch.no_grad(), autocast(dtype=torch.float16):
#                 for x,y in tqdm(val_loader, desc=f"Step{step_id}-E{epoch} val", ncols=100):
#                     x,y = x.to(device).float(), y.to(device)
#                     preds = model(x).argmax(1)
#                     vcorrect += (preds == y).sum().item()
#                     vtotal   += x.size(0)

#             vacc = vcorrect / vtotal
#             print(f"Epoch {epoch}  Val Acc={vacc:.4f}")

#             scheduler.step(vacc)
#             if vacc > best_val:
#                 best_val = vacc; stale = 0
#                 torch.save(model.state_dict(),
#                            os.path.join(model_dir, f"step{step_id}_best.pth"))
#             else:
#                 stale += 1
#             if stale >= patience:
#                 print("Early stop."); break

       
#         for cls in new_task:
#             idxs = [i for i,l in enumerate(full_train.labels) if l == cls]
#             memory_buffer[cls] = random.sample(idxs, memory_per_class)

#         teacher = copy.deepcopy(model).eval()

    
#     torch.save(model.state_dict(), os.path.join(model_dir, "final_incremental.pth"))
#     print("Incremental training complete.")


# if __name__ == "__main__":
#     train_incremental()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, random, copy, warnings
# Limit CPU threads
for var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"
):
    os.environ[var] = "4"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import CombinedSketchDataset
from model   import SketchLSTM

warnings.filterwarnings("ignore", category=UserWarning)


def train_incremental(
    data_dir: str  = r"C:\Users\khand\AppData\Local\Google\Cloud SDK\sketchrnnnpz\data\subsampledv1",
    model_dir: str = r"C:\Users\khand\AppData\Local\Google\Cloud SDK\sketchrnnnpz\models",
    base_checkpoint: str = r"C:\Users\khand\AppData\Local\Google\Cloud SDK\sketchrnnnpz\models\best_base.pth",
    task_size: int = 10,
    memory_per_class: int = 20,
    alpha: float = 1.0,
    lr: float = 5e-4,
    weight_decay: float = 1e-6,
    num_epochs: int = 15,
    patience: int = 3,
    batch_size: int = 128,
    pad_length: int = 100,
    max_grad_norm: float = 5.0,
):
    # Determine device dynamically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    os.makedirs(model_dir, exist_ok=True)

    # Load full datasets
    full_train = CombinedSketchDataset(data_dir, "train", pad_length)
    full_val   = CombinedSketchDataset(data_dir, "valid", pad_length)
    num_classes = len(full_train.files)

    # Create tasks
    tasks = [list(range(i, min(i+task_size, num_classes)))
             for i in range(0, num_classes, task_size)]

    # Initialize memory buffer for first task
    memory_buffer = {}
    for cls in tasks[0]:
        idxs = [i for i,l in enumerate(full_train.labels) if l == cls]
        memory_buffer[cls] = random.sample(idxs, memory_per_class)

    # Load base model
    model = SketchLSTM(3, 512, 2, 0.5, num_classes).to(device)
    model.load_state_dict(torch.load(base_checkpoint, map_location=device))
    teacher = copy.deepcopy(model).eval()

    criterion_ce  = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    scaler = GradScaler()

    # Incremental steps
    for step_id, new_task in enumerate(tasks[1:], start=1):
        print(f"\n=== Step {step_id}: classes {new_task} ===")

        # Combine memory indices and new class indices
        train_idxs = [i for idxs in memory_buffer.values() for i in idxs]
        for cls in new_task:
            train_idxs += [i for i,l in enumerate(full_train.labels) if l == cls]
        random.shuffle(train_idxs)

        val_idxs = [i for i,l in enumerate(full_val.labels)
                    if l in (*memory_buffer.keys(), *new_task)]

        train_loader = DataLoader(Subset(full_train, train_idxs), batch_size=batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(Subset(full_val,   val_idxs),   batch_size=batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2, verbose=True)

        best_val_acc = 0.0
        stale = 0

        for epoch in range(1, num_epochs+1):
            # Training
            model.train()
            train_loss = train_correct = train_total = 0
            for x,y in tqdm(train_loader, desc=f"Step{step_id}-Train Ep{epoch}"):
                x, y = x.to(device).float(), y.to(device)
                optimizer.zero_grad()
                with autocast():
                    logits = model(x)
                    loss_ce = criterion_ce(logits, y)
                    mask = torch.tensor([lbl in memory_buffer for lbl in y.cpu()], device=device)
                    if mask.any():
                        old_logits = teacher(x[mask].clone().detach())
                        loss_dist = criterion_mse(logits[mask], old_logits)
                    else:
                        loss_dist = torch.tensor(0., device=device)
                    loss = loss_ce + alpha * loss_dist
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * x.size(0)
                train_correct += (logits.argmax(1) == y).sum().item()
                train_total += x.size(0)

            # Validation
            model.eval()
            val_correct = val_total = 0
            with torch.no_grad(), autocast():
                for x,y in tqdm(val_loader, desc=f"Step{step_id}-Val Ep{epoch}"):
                    x, y = x.to(device).float(), y.to(device)
                    preds = model(x).argmax(1)
                    val_correct += (preds == y).sum().item()
                    val_total += y.size(0)
            val_acc = val_correct / val_total
            print(f"Epoch {epoch}  Val Acc={val_acc:.4f}")

            scheduler.step(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                stale = 0
                torch.save(model.state_dict(), os.path.join(model_dir, f"step{step_id}_best.pth"))
            else:
                stale += 1
            if stale >= patience:
                print("Early stop.")
                break

        # Update memory buffer
        for cls in new_task:
            idxs = [i for i,l in enumerate(full_train.labels) if l == cls]
            memory_buffer[cls] = random.sample(idxs, memory_per_class)
        teacher = copy.deepcopy(model).eval()

    # Final save
    torch.save(model.state_dict(), os.path.join(model_dir, "final_incremental.pth"))
    print("Incremental training complete.")

if __name__ == "__main__":
    train_incremental()
