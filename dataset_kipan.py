"""
dataset_kipan.py
================
Dataset module cho MoXGATE — KIPAN (Kidney Cancer: KICH / KIRC / KIRP).

3 phân lớp:
    KICH  (Kidney Chromophobe)      → Label 0   (~65  mẫu)
    KIRC  (Kidney Clear Cell)       → Label 1   (~352 mẫu)
    KIRP  (Kidney Papillary)        → Label 2   (~271 mẫu)
    Tổng: ~688 mẫu

Split strategy: Stratified 80/10/10 (Train/Val/Test) — giống BRCA.

Input files (từ data_final_kipan/):
    final_gene.csv                   — log2(TPM+1)
    final_mirna.csv          — log2(RPM+1)
    final_methylation.csv    — beta values [0, 1]
    final_labels.csv         — Patient ID, Cancer_Type, Target_Label
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight


# Metadata
KIPAN_SUBTYPE_NAMES = {
    0: "KICH",
    1: "KIRC",
    2: "KIRP",
}
NUM_CLASSES_KIPAN = 3


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: str):
    """Đọc 4 file final_*.csv, trả về arrays numpy đã align."""
    print(f"[Dataset KIPAN] Đọc dữ liệu từ: {data_dir}")

    gene   = pd.read_csv(os.path.join(data_dir, "final_gene.csv"),         index_col=0)
    mirna  = pd.read_csv(os.path.join(data_dir, "final_mirna.csv"),        index_col=0)
    methyl = pd.read_csv(os.path.join(data_dir, "final_methylation.csv"),  index_col=0)
    labels = pd.read_csv(os.path.join(data_dir, "final_labels.csv"),       index_col=0)

    # Align theo Patient ID
    common_ids = (set(gene.index) & set(mirna.index) &
                  set(methyl.index) & set(labels.index))
    common_ids = sorted(common_ids)

    gene   = gene.loc[common_ids]
    mirna  = mirna.loc[common_ids]
    methyl = methyl.loc[common_ids]
    labels = labels.loc[common_ids]

    print(f"[Dataset KIPAN] Gene:   {gene.shape}")
    print(f"[Dataset KIPAN] miRNA:  {mirna.shape}")
    print(f"[Dataset KIPAN] Methyl: {methyl.shape}")
    print(f"[Dataset KIPAN] Labels: {labels.shape}")

    label_arr = labels["Target_Label"].values.astype(np.int64)

    # In phân bố lớp
    for k, name in KIPAN_SUBTYPE_NAMES.items():
        count = (label_arr == k).sum()
        print(f"[Dataset KIPAN]   {name} (label={k}): {count} mẫu")

    return (
        gene.values.astype(np.float32),
        mirna.values.astype(np.float32),
        methyl.values.astype(np.float32),
        label_arr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────

class OmicsDataset(Dataset):
    def __init__(self, gene, mirna, methyl, labels, idx):
        self.gene   = torch.from_numpy(gene[idx])
        self.mirna  = torch.from_numpy(mirna[idx])
        self.methyl = torch.from_numpy(methyl[idx])
        self.labels = torch.from_numpy(labels[idx])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.gene[i], self.mirna[i], self.methyl[i], self.labels[i]


# ─────────────────────────────────────────────────────────────────────────────
# 3. STRATIFIED SPLIT (80/10/10)
# ─────────────────────────────────────────────────────────────────────────────

def split_data(labels: np.ndarray,
               val_ratio: float = 0.1,
               test_ratio: float = 0.2,
               seed: int = 42):
    """
    Stratified 80/10/10 split — đảm bảo KICH (65 mẫu) có đại diện ở cả 3 split.
    """
    np.random.seed(seed)
    all_idx = np.arange(len(labels))

    # Stage 1: Tách Test set (20%)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(sss.split(all_idx, labels))

    # Stage 2: Tách Val set từ phần còn lại
    relative_val = val_ratio / (1.0 - test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed)
    train_rel, val_rel = next(sss2.split(train_val_idx, labels[train_val_idx]))

    train_idx = train_val_idx[train_rel]
    val_idx   = train_val_idx[val_rel]

    print(f"[Dataset KIPAN] Split — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    return train_idx, val_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD DATALOADERS
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    data_dir:    str,
    batch_size:  int   = 32,
    val_ratio:   float = 0.1,
    test_ratio:  float = 0.2,
    seed:        int   = 42,
    num_workers: int   = 0,
):
    """
    Load data, split, scale, và wrap vào DataLoaders.

    Returns:
        train_loader, val_loader, test_loader,
        scalers: dict{'gene', 'mirna', 'methyl'},
        dims:    dict{'gene', 'mirna', 'methyl'},
        class_weights: np.array (3,) — cho FocalLoss
    """
    gene, mirna, methyl, labels = load_data(data_dir)

    dims = {
        'gene':   gene.shape[1],
        'mirna':  mirna.shape[1],
        'methyl': methyl.shape[1],
    }

    train_idx, val_idx, test_idx = split_data(labels, val_ratio, test_ratio, seed)

    # Class weights (balanced) từ tập train
    train_labels = labels[train_idx]
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(NUM_CLASSES_KIPAN),
        y=train_labels,
    ).astype(np.float32)

    weight_dict = {KIPAN_SUBTYPE_NAMES[i]: round(float(class_weights[i]), 4)
                   for i in range(NUM_CLASSES_KIPAN)}
    print(f"[Dataset KIPAN] Class weights: {weight_dict}")

    # Scale — fit chỉ trên train
    scalers = {}
    for name, arr in [("gene", gene), ("mirna", mirna), ("methyl", methyl)]:
        scaler = StandardScaler()
        scaler.fit(arr[train_idx])
        arr[:] = scaler.transform(arr)
        scalers[name] = scaler

    print("[Dataset KIPAN] StandardScaler fitted trên train set, applied to all splits")

    # Tạo DataLoaders
    train_ds = OmicsDataset(gene, mirna, methyl, labels, train_idx)
    val_ds   = OmicsDataset(gene, mirna, methyl, labels, val_idx)
    test_ds  = OmicsDataset(gene, mirna, methyl, labels, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=len(val_ds),
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=len(test_ds),
                              shuffle=False, num_workers=num_workers)

    n_train_b = len(train_loader)
    n_val_b   = len(val_loader)
    n_test_b  = len(test_loader)
    print(f"[Dataset KIPAN] Batches — Train: {n_train_b}, Val: {n_val_b}, Test: {n_test_b}")

    return train_loader, val_loader, test_loader, scalers, dims, class_weights
