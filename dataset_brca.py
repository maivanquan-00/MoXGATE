"""
dataset_brca.py
===============
Load và chuẩn bị dữ liệu cho MoXGATE model — dành riêng cho BRCA (Breast Cancer).

Split strategy (BRCA không có cancer type "unseen" như ESCA):
    ┌────────────────────────────────────────────────────────────────┐
    │ Tổng N samples                                                  │
    │   → Stratified split: 80% Train + 10% Val + 10% Test           │
    │   (theo label để đảm bảo mỗi subtype có mặt trong cả 3 split)  │
    └────────────────────────────────────────────────────────────────┘

Subtypes BRCA (PAM50):
    0 = LumA, 1 = LumB, 2 = Her2, 3 = Basal, 4 = Normal

Normalization (tránh data leakage):
    StandardScaler.fit()       → chỉ trên TRAIN set
    StandardScaler.transform() → áp dụng cho cả train/val/test

Input files (từ data_final_brca/):
    final_gene.csv           — (N, gene_features)  log2(TPM+1)
    final_mirna.csv          — (N, mirna_features) log2(RPM+1)
    final_methylation.csv    — (N, cpg_features)   beta [0, 1]
    final_labels.csv         — (N, 2)  Clean_Subtype, Target_Label
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight


# ─────────────────────────────────────────────────────────────────────────────
# Tên subtype BRCA (PAM50) — dùng cho classification report
# ─────────────────────────────────────────────────────────────────────────────
BRCA_SUBTYPE_NAMES = ["LumA", "LumB", "Her2", "Basal", "Normal"]
NUM_CLASSES_BRCA   = len(BRCA_SUBTYPE_NAMES)   # 5


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: str):
    """
    Đọc 4 file final_*.csv từ thư mục BRCA,
    trả về arrays numpy đã align theo Patient ID.

    Args:
        data_dir: Thư mục chứa final_*.csv (data_final_brca/)

    Returns:
        gene:     (N, gene_dim)   float32
        mirna:    (N, mirna_dim)  float32
        methyl:   (N, cpg_dim)   float32
        labels:   (N,)            int64  — Target_Label (0-4)
        subtypes: (N,)            str    — Clean_Subtype (LumA/LumB/Her2/Basal/Normal)
    """
    print("[Dataset BRCA] Đọc dữ liệu từ:", data_dir)

    gene   = pd.read_csv(os.path.join(data_dir, "final_gene.csv"),        index_col=0)
    mirna  = pd.read_csv(os.path.join(data_dir, "final_mirna.csv"),       index_col=0)
    methyl = pd.read_csv(os.path.join(data_dir, "final_methylation.csv"), index_col=0)
    labels = pd.read_csv(os.path.join(data_dir, "final_labels.csv"),      index_col=0)

    # Đảm bảo thứ tự Patient ID khớp nhau
    assert list(gene.index) == list(mirna.index) == list(methyl.index) == list(labels.index), \
        "Patient ID không khớp giữa các file — chạy lại final_process_omics.py cho BRCA"

    print(f"[Dataset BRCA] Gene:   {gene.shape}")
    print(f"[Dataset BRCA] miRNA:  {mirna.shape}")
    print(f"[Dataset BRCA] Methyl: {methyl.shape}")
    print(f"[Dataset BRCA] Labels: {labels.shape}")

    return (
        gene.values.astype(np.float32),
        mirna.values.astype(np.float32),
        methyl.values.astype(np.float32),
        labels["Target_Label"].values.astype(np.int64),
        labels["Clean_Subtype"].values,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAIN / VAL / TEST SPLIT — Stratified (không split theo loại ung thư)
# ─────────────────────────────────────────────────────────────────────────────

def split_data(
    labels:     np.ndarray,
    val_ratio:  float = 0.1,
    test_ratio: float = 0.1,
    seed:       int   = 42,
):
    """
    Chia indices thành train / val / test theo stratified random split.
    BRCA dùng stratified split (không có cancer type "unseen" như GI).

    Args:
        labels:     array int64 Target_Label — dùng cho stratified split
        val_ratio:  Tỉ lệ val trên toàn bộ tập (mặc định 0.1)
        test_ratio: Tỉ lệ test trên toàn bộ tập (mặc định 0.1)
        seed:       Random seed để reproducibility

    Returns:
        train_idx, val_idx, test_idx: numpy arrays of absolute indices
    """
    np.random.seed(seed)
    N   = len(labels)
    idx = np.arange(N)

    # Bước 1: Tách test khỏi (train+val)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    rel_train_val, rel_test = next(sss1.split(idx, labels))
    train_val_idx = idx[rel_train_val]   # absolute indices
    test_idx      = idx[rel_test]        # absolute indices

    # Bước 2: Tách val khỏi train — tỉ lệ val điều chỉnh theo phần còn lại
    adjusted_val_ratio = val_ratio / (1.0 - test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val_ratio, random_state=seed)
    rel_train, rel_val = next(sss2.split(train_val_idx, labels[train_val_idx]))
    train_idx = train_val_idx[rel_train]  # absolute indices
    val_idx   = train_val_idx[rel_val]    # absolute indices

    print(f"[Dataset BRCA] Split — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    return train_idx, val_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
# 3. FIT SCALER TRÊN TRAIN, TRANSFORM TẤT CẢ
# ─────────────────────────────────────────────────────────────────────────────

def fit_and_scale(
    gene:      np.ndarray,
    mirna:     np.ndarray,
    methyl:    np.ndarray,
    train_idx: np.ndarray,
):
    """
    Fit StandardScaler chỉ trên train set, transform cả 3 split.
    Mỗi omics có scaler riêng.

    Returns:
        gene, mirna, methyl: arrays đã scale (toàn bộ N samples)
        scalers: dict {'gene': scaler, 'mirna': scaler, 'methyl': scaler}
    """
    scalers = {}

    for name, arr in [("gene", gene), ("mirna", mirna), ("methyl", methyl)]:
        scaler = StandardScaler()
        scaler.fit(arr[train_idx])      # chỉ học từ train
        arr[:] = scaler.transform(arr)  # transform in-place toàn bộ
        scalers[name] = scaler

    print("[Dataset BRCA] StandardScaler fitted trên train set, applied to all splits")
    return gene, mirna, methyl, scalers


# ─────────────────────────────────────────────────────────────────────────────
# 4. PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class OmicsDataset(Dataset):
    """
    PyTorch Dataset cho 3-omics input (dùng chung với GI, không phụ thuộc).

    Args:
        gene:    (N, gene_dim)   numpy float32
        mirna:   (N, mirna_dim)  numpy float32
        methyl:  (N, methyl_dim) numpy float32
        labels:  (N,)            numpy int64
        indices: array of int — subset indices (train/val/test)
    """

    def __init__(self, gene, mirna, methyl, labels, indices):
        self.gene   = torch.tensor(gene[indices],   dtype=torch.float32)
        self.mirna  = torch.tensor(mirna[indices],  dtype=torch.float32)
        self.methyl = torch.tensor(methyl[indices], dtype=torch.float32)
        self.labels = torch.tensor(labels[indices], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.gene[idx],
            self.mirna[idx],
            self.methyl[idx],
            self.labels[idx],
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. HÀM TIỆN ÍCH: BUILD ALL DATALOADERS
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    data_dir:   str,
    batch_size: int   = 32,
    val_ratio:  float = 0.1,
    test_ratio: float = 0.1,
    seed:       int   = 42,
    num_workers:int   = 0,
):
    """
    Hàm all-in-one: load → split → scale → dataset → dataloader.

    Args:
        data_dir:    Thư mục data_final_brca/ (chứa final_*.csv)
        batch_size:  Batch size cho DataLoader (mặc định 32)
        val_ratio:   Tỉ lệ val trên toàn bộ tập (mặc định 0.1 = 10%)
        test_ratio:  Tỉ lệ test trên toàn bộ tập (mặc định 0.1 = 10%)
        seed:        Random seed
        num_workers: Số worker cho DataLoader

    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
        scalers:       dict scalers
        dims:          dict {'gene': int, 'mirna': int, 'methyl': int}
        class_weights: np.array (5,) — balanced class weights từ train set
    """
    # 1. Load
    gene, mirna, methyl, labels, subtypes = load_data(data_dir)

    dims = {
        'gene':   gene.shape[1],
        'mirna':  mirna.shape[1],
        'methyl': methyl.shape[1],
    }

    # 2. Split (stratified theo label)
    train_idx, val_idx, test_idx = split_data(labels, val_ratio, test_ratio, seed)

    # 3. Compute class weights từ class distribution của train set
    train_labels = labels[train_idx]
    present_classes = np.unique(train_labels)
    class_weights_partial = compute_class_weight(
        'balanced',
        classes=present_classes,
        y=train_labels,
    ).astype(np.float32)
    # Điền đủ NUM_CLASSES_BRCA phần tử (phòng trường hợp 1 class vắng mặt trong train)
    class_weights = np.ones(NUM_CLASSES_BRCA, dtype=np.float32)
    for cls, w in zip(present_classes, class_weights_partial):
        class_weights[cls] = w

    print(f"[Dataset BRCA] Class weights: {dict(zip(BRCA_SUBTYPE_NAMES, class_weights))}")

    # 4. Scale (fit chỉ trên train)
    gene, mirna, methyl, scalers = fit_and_scale(gene, mirna, methyl, train_idx)

    # 5. Dataset
    train_ds = OmicsDataset(gene, mirna, methyl, labels, train_idx)
    val_ds   = OmicsDataset(gene, mirna, methyl, labels, val_idx)
    test_ds  = OmicsDataset(gene, mirna, methyl, labels, test_idx)

    # 6. DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=len(val_ds), shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=len(test_ds), shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[Dataset BRCA] Batches — Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    return train_loader, val_loader, test_loader, scalers, dims, class_weights


# ─────────────────────────────────────────────────────────────────────────────
# 6. QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else \
        r"d:\ĐATN\MoXGATE\data_final_brca"

    train_loader, val_loader, test_loader, scalers, dims, class_weights = build_dataloaders(
        data_dir=DATA_DIR,
        batch_size=32,
    )

    # Kiểm tra 1 batch
    gene_b, mirna_b, methyl_b, label_b = next(iter(train_loader))
    print(f"\n[Sanity BRCA] Train batch — gene: {gene_b.shape}, mirna: {mirna_b.shape}, "
          f"methyl: {methyl_b.shape}, labels: {label_b.shape}")
    print(f"[Sanity BRCA] Label distribution (train batch): {label_b.bincount().tolist()}")
    print(f"[Sanity BRCA] Gene range after scale: [{gene_b.min():.2f}, {gene_b.max():.2f}]")
    print(f"[Sanity BRCA] Feature dims: {dims}")
    print(f"[Sanity BRCA] Class weights: {class_weights}")
    print("\n✓ Dataset BRCA sanity check passed!")
