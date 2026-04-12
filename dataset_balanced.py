"""
dataset_balanced.py
===================
Dataloader tích hợp WeightedRandomSampler để xử lý mất cân bằng lớp cực hạn (GI).

Mục tiêu: Ép mô hình phải "nhìn thấy" các mẫu thuộc nhóm hiếm (EBV, HM-SNV) 
nhiều hơn trong mỗi epoch thông qua Oversampling.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

# Nhãn GI (từ preprocess_labels.py): 0: CIN, 1: GS, 2: MSI, 3: HM-SNV, 4: EBV
GI_SUBTYPE_NAMES = {
    0: "CIN",
    1: "GS",
    2: "MSI",
    3: "HM-SNV",
    4: "EBV"
}

def load_data(data_dir: str):
    gene   = pd.read_csv(os.path.join(data_dir, "final_gene.csv"),         index_col=0)
    mirna  = pd.read_csv(os.path.join(data_dir, "final_mirna.csv"),        index_col=0)
    methyl = pd.read_csv(os.path.join(data_dir, "final_methylation.csv"),  index_col=0)
    labels = pd.read_csv(os.path.join(data_dir, "final_labels.csv"),       index_col=0)
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

def split_data_new(labels: np.ndarray, test_ratio: float = 0.2, val_ratio: float = 0.1, seed: int = 42):
    np.random.seed(seed)
    all_idx = np.arange(len(labels))
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(all_idx, labels))
    relative_val_ratio = val_ratio / (1.0 - test_ratio)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_ratio, random_state=seed)
    train_idx_rel, val_idx_rel = next(sss_val.split(train_val_idx, labels[train_val_idx]))
    train_idx = train_val_idx[train_idx_rel]
    val_idx   = train_val_idx[val_idx_rel]
    return train_idx, val_idx, test_idx

def build_dataloaders_balanced(
    data_dir:    str,
    batch_size:  int   = 32,
    test_ratio:  float = 0.2,
    val_ratio:   float = 0.1,
    seed:        int   = 42,
    num_workers: int   = 0,
):
    gene, mirna, methyl, labels, cancer = load_data(data_dir)
    dims = {'gene': gene.shape[1], 'mirna': mirna.shape[1], 'methyl': methyl.shape[1]}
    train_idx, val_idx, test_idx = split_data_new(labels, test_ratio, val_ratio, seed)

    # 1. Tính toán trọng số mẫu cho Sampler (Chỉ cho Train set)
    train_labels = labels[train_idx]
    class_sample_count = np.bincount(train_labels, minlength=5)
    # Trọng số của mỗi lớp = 1 / số lượng mẫu lớp đó
    weight = 1.0 / (class_sample_count + 1e-6)
    # Trọng số của từng mẫu trong train_idx
    samples_weight = torch.from_numpy(weight[train_labels])
    
    # Tạo Sampler
    sampler = WeightedRandomSampler(
        weights=samples_weight, 
        num_samples=len(samples_weight), 
        replacement=True  # Bật Oversampling
    )

    # 2. Scale
    for arr in [gene, mirna, methyl]:
        scaler = StandardScaler()
        scaler.fit(arr[train_idx])
        arr[:] = scaler.transform(arr)

    # 3. Loaders
    train_ds = OmicsDataset(gene, mirna, methyl, labels, train_idx)
    # shuffle=False khi dùng Sampler
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    
    val_ds = OmicsDataset(gene, mirna, methyl, labels, val_idx)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False, num_workers=num_workers)
    
    test_ds = OmicsDataset(gene, mirna, methyl, labels, test_idx)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, num_workers=num_workers)

    # Class weights cho Loss (vẫn nên giữ để double-check)
    class_weights = (len(train_labels) / (5 * class_sample_count)).astype(np.float32)

    print(f"[Dataset Balanced] Oversampling enabled via WeightedRandomSampler.")
    print(f"[Dataset Balanced] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return train_loader, val_loader, test_loader, dims, class_weights

if __name__ == "__main__":
    # Sanity check: đếm class trong 1 batch train
    import config
    train_loader, _, _, _, _ = build_dataloaders_balanced(config.GI_FINAL_DIR, batch_size=32)
    for _, _, _, labels in train_loader:
        print("Batch labels distribution:", np.bincount(labels.numpy(), minlength=5))
        break
