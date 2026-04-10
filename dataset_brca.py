import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

def load_data(data_dir: str):
    print("[Dataset] Đọc dữ liệu từ:", data_dir)
    gene   = pd.read_csv(os.path.join(data_dir, "final_gene.csv"),         index_col=0)
    mirna  = pd.read_csv(os.path.join(data_dir, "final_mirna.csv"),        index_col=0)
    methyl = pd.read_csv(os.path.join(data_dir, "final_methylation.csv"),  index_col=0)
    labels = pd.read_csv(os.path.join(data_dir, "final_labels.csv"),       index_col=0)
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
        labels["Clean_Subtype"].values,
    )

class OmicsDataset(Dataset):
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

def build_dataloaders(
    data_dir:   str,
    batch_size: int   = 32,
    val_ratio:  float = 0.1,
    test_ratio: float = 0.1,
    seed:       int   = 42,
    num_workers:int   = 0,
):
    gene, mirna, methyl, labels, subtypes = load_data(data_dir)
    dims = {
        'gene':   gene.shape[1],
        'mirna':  mirna.shape[1],
        'methyl': methyl.shape[1],
    }
    N = len(labels)
    np.random.seed(seed)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    idx = np.arange(N)
    train_val_idx, test_idx = next(sss1.split(idx, labels))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio/(1-test_ratio), random_state=seed)
    train_idx, val_idx = next(sss2.split(train_val_idx, labels[train_val_idx]))
    print(f"[Dataset] Split — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    gene, mirna, methyl, scalers = fit_and_scale(gene, mirna, methyl, train_idx)
    train_ds = OmicsDataset(gene, mirna, methyl, labels, train_idx)
    val_ds   = OmicsDataset(gene, mirna, methyl, labels, val_idx)
    test_ds  = OmicsDataset(gene, mirna, methyl, labels, test_idx)
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
    print(f"[Dataset] Batches — Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    return train_loader, val_loader, test_loader, scalers, dims

def fit_and_scale(gene, mirna, methyl, train_idx):
    scalers = {}
    for name, arr in [("gene", gene), ("mirna", mirna), ("methyl", methyl)]:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(arr[train_idx])
        arr[:] = scaler.transform(arr)
        scalers[name] = scaler
    print("[Dataset] StandardScaler fitted trên train set, applied to all splits")
    return gene, mirna, methyl, scalers
