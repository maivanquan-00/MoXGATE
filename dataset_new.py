"""
dataset_new.py
==============
Load và chuẩn bị dữ liệu cho MoXGATE với CHIẾN LƯỢC CHIA DỮ LIỆU MỚI (GI).

Thay đổi so với dataset.py bản gốc:
    - Gộp toàn bộ samples (COAD, READ, STAD, ESCA) vào một pool duy nhất.
    - Sử dụng StratifiedSplit (80% TrainVal, 20% Test) để đảm bảo tập test 
      đại diện đầy đủ cho các phân lớp hiếm (EBV, HM-SNV).
    - Tự động hóa việc chia tỉ lệ Train/Val/Test (80/10/10).
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

# Nhãn GI (từ preprocess_labels.py): 0: CIN, 1: GS, 2: MSI, 3: HM-SNV, 4: EBV
GI_SUBTYPE_NAMES = {
    0: "CIN",
    1: "GS",
    2: "MSI",
    3: "HM-SNV",
    4: "EBV"
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: str):
    """Đọc 4 file final_*.csv, trả về arrays numpy."""
    gene   = pd.read_csv(os.path.join(data_dir, "final_gene_symbol.csv"),         index_col=0)
    mirna  = pd.read_csv(os.path.join(data_dir, "final_mirna.csv"),        index_col=0)
    methyl = pd.read_csv(os.path.join(data_dir, "final_methylation.csv"),  index_col=0)
    labels = pd.read_csv(os.path.join(data_dir, "final_labels.csv"),       index_col=0)

    assert list(gene.index) == list(mirna.index) == list(methyl.index) == list(labels.index)

    return (
        gene.values.astype(np.float32),
        mirna.values.astype(np.float32),
        methyl.values.astype(np.float32),
        labels["Target_Label"].values.astype(np.int64),
        labels["Cancer_Type"].values,
    )

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
# 2. MODERNIZED SPLIT STRATEGY (80/10/10 Stratified)
# ─────────────────────────────────────────────────────────────────────────────

def split_data_new(labels: np.ndarray, test_ratio: float = 0.2, val_ratio: float = 0.1, seed: int = 42):
    """
    Chia 80/10/10 theo chiến lược Stratified Random (toàn bộ pool).
    """
    np.random.seed(seed)
    all_idx = np.arange(len(labels))

    # Stage 1: Tách Test set (20%)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(all_idx, labels))

    # Stage 2: Tách Val set từ phần còn lại
    # Nếu test=0.2, val=0.1 trên tổng -> val/train_val = 0.1 / 0.8 = 0.125
    relative_val_ratio = val_ratio / (1.0 - test_ratio)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_ratio, random_state=seed)
    train_idx_rel, val_idx_rel = next(sss_val.split(train_val_idx, labels[train_val_idx]))

    train_idx = train_val_idx[train_idx_rel]
    val_idx   = train_val_idx[val_idx_rel]

    print(f"[Dataset New] Split — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    return train_idx, val_idx, test_idx

# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD DATALOADERS
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders_new(
    data_dir:    str,
    batch_size:  int   = 32,
    test_ratio:  float = 0.2,
    val_ratio:   float = 0.1,
    seed:        int   = 42,
    num_workers: int   = 0,
):
    gene, mirna, methyl, labels, cancer = load_data(data_dir)
    dims = {'gene': gene.shape[1], 'mirna': mirna.shape[1], 'methyl': methyl.shape[1]}

    # Chia dữ liệu kiểu mới
    train_idx, val_idx, test_idx = split_data_new(labels, test_ratio, val_ratio, seed)

    # Class weights từ train set (rất quan trọng cho GI hiếm)
    train_labels = labels[train_idx]
    # np.bincount đếm các class hiện có (0-4)
    counts = np.bincount(train_labels, minlength=5)
    total = len(train_labels)
    # weights = total / (num_classes * counts)
    class_weights = (total / (5 * counts)).astype(np.float32)

    # Scale (fit chỉ train)
    scalers = {}
    for name, arr in [("gene", gene), ("mirna", mirna), ("methyl", methyl)]:
        scaler = StandardScaler()
        scaler.fit(arr[train_idx])
        arr[:] = scaler.transform(arr)
        scalers[name] = scaler

    # Pack
    loaders = []
    for idx, shuffle, label in [(train_idx, True, "Train"), (val_idx, False, "Val"), (test_idx, False, "Test")]:
        ds = OmicsDataset(gene, mirna, methyl, labels, idx)
        bs = batch_size if label == "Train" else len(ds)
        loaders.append(DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers))

    return loaders[0], loaders[1], loaders[2], scalers, dims, class_weights

if __name__ == "__main__":
    # Test distribution
    import config
    loaders = build_dataloaders_new(config.GI_FINAL_DIR)
    print("✓ Dataset New initialization successful.")
