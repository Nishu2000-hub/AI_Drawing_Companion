import os
import numpy as np
import torch
from torch.utils.data import Dataset

def apply_pca_rotation(seq):
    # seq: (L,3) relative deltas + pen flag
    coords = np.cumsum(seq[:, :2], axis=0)        # absolute positions
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov    = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    angle = np.arctan2(principal[1], principal[0])
    # rotate by -angle to align principal axis with x‑axis
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    rotated = centered.dot(R) + mean
    # back to relative deltas
    deltas = np.diff(np.vstack((mean, rotated)), axis=0)
    return np.concatenate((deltas, seq[:, 2:3]), axis=1)


def preprocess_sequence(seq, pad_length=100):
    # seq: numpy array (L,3) with ints
    seq = seq.astype(np.float32)
    seq = apply_pca_rotation(seq)                # orientation invariance
    # dynamic scaling
    max_delta = np.abs(seq[:, :2]).max() or 1.0
    seq[:, :2] /= max_delta
    # pad or truncate to fixed length
    L = seq.shape[0]
    if L >= pad_length:
        seq = seq[:pad_length]
    else:
        pad_cnt = pad_length - L
        pad = np.zeros((pad_cnt, 3), dtype=np.float32)
        pad[:, 2] = 1.0  # pen‑up for padding
        seq = np.vstack((seq, pad))
    return seq


class CombinedSketchDataset(Dataset):
    def __init__(self, data_dir, split='train', pad_length=100):
        """
        data_dir: path to subsampled .npz files (50 categories)
        split: 'train', 'valid', or 'test'
        pad_length: fixed sequence length after padding
        """
        self.split      = split
        self.pad_length = pad_length

        # 1) find all .npz files
        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith('.npz')
        ])

        # 2) load each split into memory (fix_imports & latin1 for unpickling)
        self.data = []
        self.labels = []
        for label, path in enumerate(self.files):
            npz = np.load(path, allow_pickle=True,
                          fix_imports=True, encoding='latin1')
            arr = npz[split]
            for seq in arr:
                self.data.append(seq)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_seq = self.data[idx]
        x = preprocess_sequence(raw_seq, pad_length=self.pad_length).astype(np.float32)
        y = self.labels[idx]
        return torch.from_numpy(x), y
