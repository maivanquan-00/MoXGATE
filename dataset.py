"""
dataset.py
==========
Load và chuẩn bị dữ liệu cho MoXGATE model.

Split strategy (theo Appendix B của paper):
    ┌─────────────────────────────────────────────────────────────────┐
    │ ESCA (79 samples)          → Test set (unseen cancer type)      │
    │ COAD + READ + STAD (838)   → 90% Train (754) + 10% Val (84)     │
    └─────────────────────────────────────────────────────────────────┘

Normalization (tránh data leakage):
    StandardScaler.fit()       → chỉ trên TRAIN set
    StandardScaler.transform() → áp dụng cho cả train/val/test

Input files (từ data_final/):
    final_gene.csv           — (917, 19962) log2(TPM+1)
    final_mirna.csv          — (917, 1881)  log2(RPM+1)
    final_methylation.csv    — (917, 23111) beta [0, 1]
    final_labels.csv         — (917, 3)     Patient ID → Cancer_Type, Clean_Subtype, Target_Label
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: str):
    """
    Đọc 4 file final_*.csv, trả về arrays numpy đã align theo Patient ID.

    Args:
        data_dir: Thư mục chứa final_*.csv (data_final/)

    Returns:
        gene:    (N, 19962) float32
        mirna:   (N, 1881)  float32
        methyl:  (N, 23111) float32
        labels:  (N,)       int64   — Target_Label (0-4)
        cancer:  (N,)       str     — Cancer_Type (COAD/ESCA/READ/STAD)
    """
    print("[Dataset] Đọc dữ liệu từ:", data_dir)

    gene   = pd.read_csv(os.path.join(data_dir, "final_gene.csv"),         index_col=0)
    mirna  = pd.read_csv(os.path.join(data_dir, "final_mirna.csv"),        index_col=0)
    methyl = pd.read_csv(os.path.join(data_dir, "final_methylation.csv"),  index_col=0)
    labels = pd.read_csv(os.path.join(data_dir, "final_labels.csv"),       index_col=0)

    # Đảm bảo thứ tự Patient ID khớp nhau
    assert list(gene.index) == list(mirna.index) == list(methyl.index) == list(labels.index), \
        "Patient ID không khớp giữa các file — chạy lại final_process_omics.py"

    print(f"[Dataset] Gene:   {gene.shape}")
    print(f"[Dataset] miRNA:  {mirna.shape}")
    print(f"[Dataset] Methyl: {methyl.shape}")
    print(f"[Dataset] Labels: {labels.shape}")

    return (
        gene.values.astype(np.float32),
        mirna.values.astype(np.float32),
        methyl.values.astype(np.float32),
        labels["Target_Label"].values.astype(np.int64),
        labels["Cancer_Type"].values,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_data(cancer_types: np.ndarray, val_ratio: float = 0.1, seed: int = 42):
    """
    Chia indices thành train / val / test theo strategy của paper.

    Test  = ESCA (fixed — không random)
    Train/Val = COAD+READ+STAD, chia 90-10 ngẫu nhiên (seed cố định)

    Args:
        cancer_types: array string Cancer_Type cho mỗi sample
        val_ratio:    Tỉ lệ validation từ tập train (mặc định 0.1)
        seed:         Random seed để reproducibility

    Returns:
        train_idx, val_idx, test_idx: numpy arrays of indices
    """
    np.random.seed(seed)

    all_idx  = np.arange(len(cancer_types))
    test_idx = all_idx[cancer_types == "ESCA"]
    train_val_idx = all_idx[cancer_types != "ESCA"]

    # Shuffle train+val
    shuffled = train_val_idx.copy()
    np.random.shuffle(shuffled)

    n_val     = max(1, int(len(shuffled) * val_ratio))
    val_idx   = shuffled[:n_val]
    train_idx = shuffled[n_val:]

    print(f"[Dataset] Split — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
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
        scaler.fit(arr[train_idx])        # chỉ học từ train
        arr[:] = scaler.transform(arr)    # transform in-place toàn bộ
        scalers[name] = scaler

    print("[Dataset] StandardScaler fitted trên train set, applied to all splits")
    return gene, mirna, methyl, scalers


# ─────────────────────────────────────────────────────────────────────────────
# 4. PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class OmicsDataset(Dataset):
    """
    PyTorch Dataset cho 3-omics input.

    Args:
        gene:    (N, gene_dim)   numpy float32
        mirna:   (N, mirna_dim)  numpy float32
        methyl:  (N, methyl_dim) numpy float32
        labels:  (N,)            numpy int64
        indices: array of int — subset indices (train/val/test)
    """

    def __init__(
        self,
        gene:    np.ndarray,
        mirna:   np.ndarray,
        methyl:  np.ndarray,
        labels:  np.ndarray,
        indices: np.ndarray,
    ):
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
    seed:       int   = 42,
    num_workers:int   = 0,
):
    """
    Hàm all-in-one: load → split → scale → dataset → dataloader.

    Args:
        data_dir:    Thư mục data_final/
        batch_size:  Batch size (chỉ áp dụng cho val/test; train luôn full-batch theo paper)
        val_ratio:   Tỉ lệ val từ tập COAD+READ+STAD (mặc định 0.1)
        seed:        Random seed
        num_workers: Số worker cho DataLoader

    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
        scalers: dict scalers (cần lưu lại nếu muốn inference sau)
        dims: dict {'gene': int, 'mirna': int, 'methyl': int}
    """
    # 1. Load
    gene, mirna, methyl, labels, cancer = load_data(data_dir)

    dims = {
        'gene':   gene.shape[1],
        'mirna':  mirna.shape[1],
        'methyl': methyl.shape[1],
    }

    # 2. Split
    train_idx, val_idx, test_idx = split_data(cancer, val_ratio, seed)

    # 3. Scale (fit chỉ trên train)
    gene, mirna, methyl, scalers = fit_and_scale(gene, mirna, methyl, train_idx)

    # 4. Dataset
    train_ds = OmicsDataset(gene, mirna, methyl, labels, train_idx)
    val_ds   = OmicsDataset(gene, mirna, methyl, labels, val_idx)
    test_ds  = OmicsDataset(gene, mirna, methyl, labels, test_idx)

    # 5. DataLoader
    #    Paper dùng cross-sample self-attention (Eq. 4: A ∈ R^{N×N})
    #    → cần full-batch để tất cả samples attend vào nhau nhất quán
    #    giữa train và eval (val/test đã dùng full-batch sẵn).
    train_loader = DataLoader(
        train_ds, batch_size=len(train_ds), shuffle=True,
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

    print(f"[Dataset] Batches — Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    return train_loader, val_loader, test_loader, scalers, dims


# ─────────────────────────────────────────────────────────────────────────────
# 6. QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else \
        "/content/drive/MyDrive/ĐATN_2025.2/data_final"

    train_loader, val_loader, test_loader, scalers, dims = build_dataloaders(
        data_dir=DATA_DIR,
        batch_size=32,
    )

    # Kiểm tra 1 batch
    gene_b, mirna_b, methyl_b, label_b = next(iter(train_loader))
    print(f"\n[Sanity] Train batch — gene: {gene_b.shape}, mirna: {mirna_b.shape}, "
          f"methyl: {methyl_b.shape}, labels: {label_b.shape}")
    print(f"[Sanity] Label distribution (train batch): {label_b.bincount().tolist()}")
    print(f"[Sanity] Gene range after scale: [{gene_b.min():.2f}, {gene_b.max():.2f}]")
    print(f"[Sanity] Feature dims: {dims}")
    print("\n✓ Dataset sanity check passed!")
