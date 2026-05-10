"""
train_kfold.py
==============
Script chạy 5-Fold Cross Validation trên nhiều Global Seeds cho mô hình MoXGATE.
Áp dụng chung cho tất cả dataset (GI, BRCA, KIPAN, UCEC, v.v.).

Cơ chế:
  - Outer loop: Duyệt qua danh sách Seeds (ví dụ: 42, 123, 2024).
  - K-Fold: Dùng StratifiedKFold (n_splits=5, random_state=seed).
    -> Đảm bảo cách chia fold CÙNG splits qua các mô hình khác (nếu dùng chung seed).
  - Validation: Cắt 10% từ tập Train của fold đó để làm Early Stopping.
  - Scaling: Fit StandardScaler trên TẬP TRAIN, transform Train/Val/Test.
  - Metrics: Ưu tiên in và lưu F1-Macro, sau đó đến F1-Weighted và Accuracy.
"""

import os
import argparse
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

import config
from model import MoXGATE

class OmicsDataset(Dataset):
    def __init__(self, gene, mirna, methyl, labels, indices=None):
        if indices is not None:
            self.gene   = torch.tensor(gene[indices],   dtype=torch.float32)
            self.mirna  = torch.tensor(mirna[indices],  dtype=torch.float32)
            self.methyl = torch.tensor(methyl[indices], dtype=torch.float32)
            self.labels = torch.tensor(labels[indices], dtype=torch.long)
        else:
            self.gene   = torch.tensor(gene,   dtype=torch.float32)
            self.mirna  = torch.tensor(mirna,  dtype=torch.float32)
            self.methyl = torch.tensor(methyl, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.gene[idx], self.mirna[idx], self.methyl[idx], self.labels[idx]

def load_all_data(data_dir):
    print(f"\n[Data] Loading from: {data_dir}")
    gene   = pd.read_csv(os.path.join(data_dir, "final_gene.csv"), index_col=0)
    mirna  = pd.read_csv(os.path.join(data_dir, "final_mirna.csv"), index_col=0)
    methyl = pd.read_csv(os.path.join(data_dir, "final_methylation.csv"), index_col=0)
    labels = pd.read_csv(os.path.join(data_dir, "final_labels.csv"), index_col=0)

    # Đảm bảo index khớp
    assert list(gene.index) == list(mirna.index) == list(methyl.index) == list(labels.index), "Index mismatch!"
    # Sanity checks for common data problems (duplicates / identical rows)
    if gene.index.duplicated().any():
        print("[Warning] Duplicate Patient IDs found in final_gene.csv")
    if mirna.index.duplicated().any():
        print("[Warning] Duplicate Patient IDs found in final_mirna.csv")
    if methyl.index.duplicated().any():
        print("[Warning] Duplicate Patient IDs found in final_methylation.csv")

    dup_gene_rows = int(gene.duplicated().sum())
    dup_mirna_rows = int(mirna.duplicated().sum())
    dup_methyl_rows = int(methyl.duplicated().sum())
    if dup_gene_rows or dup_mirna_rows or dup_methyl_rows:
        print(f"[Warning] Duplicate feature rows detected - gene:{dup_gene_rows}, mirna:{dup_mirna_rows}, methyl:{dup_methyl_rows}")

    # Print label distribution for quick inspection
    if "Target_Label" in labels.columns:
        label_counts = labels["Target_Label"].value_counts().to_dict()
        print(f"[Data] Label counts: {label_counts}")
    else:
        print("[Warning] final_labels.csv missing 'Target_Label' column")

    X_gene = gene.values.astype(np.float32)
    X_mirna = mirna.values.astype(np.float32)
    X_methyl = methyl.values.astype(np.float32)
    y = labels["Target_Label"].values.astype(np.int64)
    
    dims = {
        "gene": X_gene.shape[1],
        "mirna": X_mirna.shape[1],
        "methyl": X_methyl.shape[1]
    }
    
    print(f"[Data] Samples: {len(y)} | Gene_Dim: {dims['gene']} | MiRNA_Dim: {dims['mirna']} | Methyl_Dim: {dims['methyl']}")
    return X_gene, X_mirna, X_methyl, y, dims

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets, total_loss = [], [], 0.0

    for gene, mirna, methyl, labels in loader:
        gene, mirna, methyl, labels = gene.to(device), mirna.to(device), methyl.to(device), labels.to(device)
        logits, w = model(gene, mirna, methyl)
        loss, _   = model.compute_loss(logits, labels, w)

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    metrics = {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(all_targets, all_preds),
        "macro_f1": f1_score(all_targets, all_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_targets, all_preds, average="weighted", zero_division=0),
        "macro_precision": precision_score(all_targets, all_preds, average="macro", zero_division=0),
        "weighted_precision": precision_score(all_targets, all_preds, average="weighted", zero_division=0),
        "macro_recall": recall_score(all_targets, all_preds, average="macro", zero_division=0),
        "weighted_recall": recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    }
    return metrics, all_targets, all_preds

def train_fold(seed, fold, train_idx, val_idx, test_idx, data, dims, args, device):
    X_g, X_mi, X_me, y = data
    
    # 1. StandardScaler (chỉ fit trên train)
    scaler_g  = StandardScaler().fit(X_g[train_idx])
    scaler_mi = StandardScaler().fit(X_mi[train_idx])
    scaler_me = StandardScaler().fit(X_me[train_idx])
    
    X_g_scaled  = scaler_g.transform(X_g)
    X_mi_scaled = scaler_mi.transform(X_mi)
    X_me_scaled = scaler_me.transform(X_me)
    
    # 2. Dataset & DataLoader
    train_loader = DataLoader(OmicsDataset(X_g_scaled, X_mi_scaled, X_me_scaled, y, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(OmicsDataset(X_g_scaled, X_mi_scaled, X_me_scaled, y, val_idx), batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(OmicsDataset(X_g_scaled, X_mi_scaled, X_me_scaled, y, test_idx), batch_size=args.batch_size, shuffle=False)
    
    # 3. Model init (seed phải khác per-fold để tránh artificial perfect scores)
    fold_seed = seed + fold * 1000  # Ensure different seed per fold
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)
    
    model = MoXGATE(gene_dim=dims["gene"], mirna_dim=dims["mirna"], methyl_dim=dims["methyl"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0
    epoch_logs = []
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for gene, mirna, methyl, labels in train_loader:
            gene, mirna, methyl, labels = gene.to(device), mirna.to(device), methyl.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, w = model(gene, mirna, methyl)
            loss, _ = model.compute_loss(logits, labels, w, lambda1=args.lambda1, lambda2=args.lambda2)
            train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        train_loss /= len(train_loader)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "patience": patience_counter
        })
        
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_weights = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break
                
    # 4. Evaluate Test Set
    model.load_state_dict(best_weights)
    test_metrics, test_targets, test_preds = evaluate(model, test_loader, device)
    
    # Log: how many epochs trained, best val loss
    best_epoch = epoch_logs.index(min(epoch_logs, key=lambda x: x["val_loss"])) + 1 if epoch_logs else 1
    
    return test_metrics, best_epoch, len(epoch_logs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data_final_xxx")
    parser.add_argument("--save_path", type=str, default="kfold_results.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lambda1", type=float, default=0.01)
    parser.add_argument("--lambda2", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED, help="Seed for 5-fold split and training")
    parser.add_argument("--multi_seed", action="store_true", help="Run legacy 3-seed mode (42, 123, 2024)")
    parser.add_argument("--test_mode", action="store_true", help="Chạy 1 fold cho mục đích test code nhanh")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Cập nhật đường dẫn tự động qua config (hỗ trợ Colab)
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(config.BASE_DIR, args.data_dir)
    if not os.path.isabs(args.save_path):
        args.save_path = os.path.join(config.BASE_DIR, args.save_path)

    # Load dataset gộp cứng (GIAC, BRCA, KIPAN...) như nhau
    data_tuple = load_all_data(args.data_dir)
    _, _, _, y, dims = data_tuple
    
    seeds = [42, 123, 2024] if args.multi_seed else [args.seed]
    print(f"[Run Mode] Seeds: {seeds}")
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*50}\nRUNNING SEED: {seed}\n{'='*50}")
        # Cố định random splits như các model khác
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        seed_metrics = []
        for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
            print(f"  -> Fold {fold}/5 ... ", end="")
            
            # Tách Val (10% của đoạn train_val_idx)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
            train_rel_idx, val_rel_idx = next(sss.split(train_val_idx, y[train_val_idx]))
            
            train_idx = train_val_idx[train_rel_idx]
            val_idx   = train_val_idx[val_rel_idx]
            # Sanity checks: ensure splits are disjoint and print label distributions
            if len(set(train_idx).intersection(set(test_idx))) != 0:
                raise AssertionError("Train/Test overlap detected!")
            if len(set(train_idx).intersection(set(val_idx))) != 0:
                raise AssertionError("Train/Val overlap detected!")
            if len(set(val_idx).intersection(set(test_idx))) != 0:
                raise AssertionError("Val/Test overlap detected!")

            try:
                print(f"    Labels — train: {np.bincount(y[train_idx])}, val: {np.bincount(y[val_idx])}, test: {np.bincount(y[test_idx])}")
            except Exception:
                # Fallback if labels are not small integers starting at 0
                print(f"    Labels sizes — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

            test_metrics, best_epoch, total_epochs = train_fold(seed, fold, train_idx, val_idx, test_idx, data_tuple[:-1], dims, args, device)
            
            print(f"Test F1-Macro: {test_metrics['macro_f1']:.4f} | F1-Weighted: {test_metrics['weighted_f1']:.4f} | Accuracy: {test_metrics['accuracy']:.4f} | Precision-Macro: {test_metrics['macro_precision']:.4f} | Recall-Macro: {test_metrics['macro_recall']:.4f}")
            print(f"    Training stopped at epoch {best_epoch}/{total_epochs}")
            seed_metrics.append(test_metrics)
            
            if args.test_mode:  # Dành cho user check code 1-2 fold (tiết kiệm compute)
                print("    [!] TEST MODE DỪNG SỚM.")
                break
        
        avg_macro = np.mean([x["macro_f1"] for x in seed_metrics])
        std_macro = np.std([x["macro_f1"] for x in seed_metrics])
        avg_weight = np.mean([x["weighted_f1"] for x in seed_metrics])
        std_weight = np.std([x["weighted_f1"] for x in seed_metrics])
        avg_acc = np.mean([x["accuracy"] for x in seed_metrics])
        std_acc = np.std([x["accuracy"] for x in seed_metrics])
        avg_pre = np.mean([x["macro_precision"] for x in seed_metrics])
        std_pre = np.std([x["macro_precision"] for x in seed_metrics])
        avg_rec = np.mean([x["macro_recall"] for x in seed_metrics])
        std_rec = np.std([x["macro_recall"] for x in seed_metrics])
        
        print(f"\n>>> SEED {seed} FINAL METRICS (Mean ± Std) - SOFTMAX <<<")
        print(f"{'='*80}")
        print(f"{'Metric':<20} | {'Macro':<25} | {'Weighted':<25}")
        print(f"{'-'*80}")
        print(f"{'Accuracy':<20} | {avg_acc:.4f} ± {std_acc:.4f}     | {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"{'F1-Score':<20} | {avg_macro:.4f} ± {std_macro:.4f}     | {avg_weight:.4f} ± {std_weight:.4f}")
        print(f"{'Precision':<20} | {avg_pre:.4f} ± {std_pre:.4f}     | {np.mean([x['weighted_precision'] for x in seed_metrics]):.4f} ± {np.std([x['weighted_precision'] for x in seed_metrics]):.4f}")
        print(f"{'Recall':<20} | {avg_rec:.4f} ± {std_rec:.4f}     | {np.mean([x['weighted_recall'] for x in seed_metrics]):.4f} ± {np.std([x['weighted_recall'] for x in seed_metrics]):.4f}")
        print(f"{'='*80}")
        
        all_results.append({
            "seed": seed,
            "folds": seed_metrics,
            "avg_macro_f1": avg_macro, "std_macro_f1": std_macro,
            "avg_weighted_f1": avg_weight, "std_weighted_f1": std_weight,
            "avg_accuracy": avg_acc, "std_accuracy": std_acc,
            "avg_macro_precision": avg_pre, "std_macro_precision": std_pre,
            "avg_macro_recall": avg_rec, "std_macro_recall": std_rec
        })
        
        if args.test_mode:
            break

    # Final Overall
    all_folds_metrics = [f for r in all_results for f in r["folds"]]
    print(f"\n{'*'*50}")
    print(f"HOÀN TẤT TẤT CẢ SEEDS ({len(all_folds_metrics)} RUNS)")
    ov_mac = np.mean([x["macro_f1"] for x in all_folds_metrics])
    sd_mac = np.std([x["macro_f1"] for x in all_folds_metrics])
    
    ov_wei = np.mean([x["weighted_f1"] for x in all_folds_metrics])
    sd_wei = np.std([x["weighted_f1"] for x in all_folds_metrics])
    
    ov_acc = np.mean([x["accuracy"] for x in all_folds_metrics])
    sd_acc = np.std([x["accuracy"] for x in all_folds_metrics])
    
    ov_pre = np.mean([x["macro_precision"] for x in all_folds_metrics])
    sd_pre = np.std([x["macro_precision"] for x in all_folds_metrics])
    
    ov_rec = np.mean([x["macro_recall"] for x in all_folds_metrics])
    sd_rec = np.std([x["macro_recall"] for x in all_folds_metrics])

    ov_pre_w = np.mean([x["weighted_precision"] for x in all_folds_metrics])
    sd_pre_w = np.std([x["weighted_precision"] for x in all_folds_metrics])
    
    ov_rec_w = np.mean([x["weighted_recall"] for x in all_folds_metrics])
    sd_rec_w = np.std([x["weighted_recall"] for x in all_folds_metrics])

    print("\nOVERALL FINAL METRICS (Mean ± Std) - SOFTMAX:")
    print(f"{'='*80}")
    print(f"{'Metric':<20} | {'Macro':<25} | {'Weighted':<25}")
    print(f"{'-'*80}")
    print(f"{'Accuracy':<20} | {ov_acc:.4f} ± {sd_acc:.4f}     | {ov_acc:.4f} ± {sd_acc:.4f}")
    print(f"{'F1-Score':<20} | {ov_mac:.4f} ± {sd_mac:.4f}     | {ov_wei:.4f} ± {sd_wei:.4f}")
    print(f"{'Precision':<20} | {ov_pre:.4f} ± {sd_pre:.4f}     | {ov_pre_w:.4f} ± {sd_pre_w:.4f}")
    print(f"{'Recall':<20} | {ov_rec:.4f} ± {sd_rec:.4f}     | {ov_rec_w:.4f} ± {sd_rec_w:.4f}")
    print(f"{'='*80}")
    
    with open(args.save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {args.save_path}")

if __name__ == "__main__":
    main()
