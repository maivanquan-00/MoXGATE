"""
train.py
========
Training script cho MoXGATE model.

Hyperparameters (theo paper Section 2.1.6):
    optimizer:   AdamW, lr=1e-4, weight_decay=1e-2
    batch_size:  32
    epochs:      100 (early stopping patience=15)
    focal:       γ=2, α=1 (paper Section 2.1.4)
    λ1:          0.01  (modality weight regularization)
    λ2:          1e-4  (Frobenius norm cross-attention — paper không chỉ định)

Cách chạy trên Colab:
    !python MoXGATE/train.py \\
        --data_dir  "/content/drive/MyDrive/ĐATN_2025.2/data_final" \\
        --save_dir  "/content/drive/MyDrive/ĐATN_2025.2/checkpoints"

    # Với custom hyperparams:
    !python MoXGATE/train.py \\
        --data_dir  "/content/drive/MyDrive/ĐATN_2025.2/data_final" \\
        --save_dir  "/content/drive/MyDrive/ĐATN_2025.2/checkpoints" \\
        --epochs 150 --batch_size 64 --lr 5e-5
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from dataset import build_dataloaders
from model   import MoXGATE


# ─────────────────────────────────────────────────────────────────────────────
# 1. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    """
    Chạy inference trên 1 DataLoader, trả về loss + metrics.

    Returns:
        metrics: dict với keys accuracy, f1, precision, recall, loss
        all_preds, all_targets: numpy arrays để dùng cho classification_report
    """
    model.eval()
    all_preds, all_targets, total_loss = [], [], 0.0

    for gene, mirna, methyl, labels in loader:
        gene   = gene.to(device)
        mirna  = mirna.to(device)
        methyl = methyl.to(device)
        labels = labels.to(device)

        logits, w = model(gene, mirna, methyl)
        loss, _   = model.compute_loss(logits, labels, w)

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    metrics = {
        "loss":      total_loss / len(loader),
        "accuracy":  accuracy_score(all_targets, all_preds),
        "f1":        f1_score(all_targets, all_preds, average="weighted", zero_division=0),
        "precision": precision_score(all_targets, all_preds, average="weighted", zero_division=0),
        "recall":    recall_score(all_targets, all_preds, average="weighted", zero_division=0),
    }
    return metrics, all_preds, all_targets


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── Setup ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'★'*60}")
    print("  MoXGATE — TRAINING")
    print(f"{'★'*60}")
    print(f"  Device     : {device}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Save dir   : {args.save_dir}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  Seed       : {args.seed}")
    print(f"{'★'*60}\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, scalers, dims, class_weights = build_dataloaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        val_ratio   = args.val_ratio,
        seed        = args.seed,
        num_workers = args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MoXGATE(
        gene_dim   = dims["gene"],
        mirna_dim  = dims["mirna"],
        methyl_dim = dims["methyl"],
    ).to(device)

    # Set per-class weights cho focal loss
    # Paper: αi = 1 cho tất cả class (Section 2.1.4)
    # Giữ α=1.0 — class_weights được tính sẵn để dùng khi cần
    # model.set_class_weights(class_weights)

    print(f"[Train] Trainable parameters: {model.count_parameters():,}\n")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # LR scheduler: giảm LR khi val loss không cải thiện
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss   = float("inf")
    best_val_acc    = 0.0
    patience_counter = 0
    history         = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss  = 0.0
        epoch_focal = 0.0
        t0 = time.time()

        for gene, mirna, methyl, labels in train_loader:
            gene   = gene.to(device)
            mirna  = mirna.to(device)
            methyl = methyl.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, w = model(gene, mirna, methyl)
            loss, focal = model.compute_loss(
                logits, labels, w,
                lambda1=args.lambda1,
                lambda2=args.lambda2,
            )
            loss.backward()

            # Gradient clipping để tránh exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss  += loss.item()
            epoch_focal += focal.item()

        # ── Validation ────────────────────────────────────────────────────────
        val_metrics, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["loss"])

        avg_train_loss  = epoch_loss  / len(train_loader)
        avg_focal_loss  = epoch_focal / len(train_loader)
        elapsed         = time.time() - t0

        w_dict = model.get_modality_weights()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} (focal: {avg_focal_loss:.4f}) | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Weights: G={w_dict['Gene']:.3f} M={w_dict['miRNA']:.3f} C={w_dict['Methylation']:.3f} | "
            f"{elapsed:.1f}s"
        )

        # Lưu history
        history.append({
            "epoch":      epoch,
            "train_loss": avg_train_loss,
            "focal_loss": avg_focal_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"w_{k}": v for k, v in w_dict.items()},
        })

        # ── Checkpoint tốt nhất (theo val accuracy) ───────────────────────────
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc  = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc":     best_val_acc,
                "val_loss":    best_val_loss,
                "config":      model.config,
                "dims":        dims,
            }, os.path.join(args.save_dir, "best_model.pt"))
            print(f"  ✓ Saved best model (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Train] Early stopping tại epoch {epoch} (patience={args.patience})")
                break

    # ── Lưu history ───────────────────────────────────────────────────────────
    history_path = os.path.join(args.save_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Train] History đã lưu: {history_path}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  ĐÁNH GIÁ TRÊN TEST SET (ESCA)")
    print(f"{'─'*60}")

    checkpoint = torch.load(os.path.join(args.save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_metrics, test_preds, test_targets = evaluate(model, test_loader, device)

    subtype_names = ["CIN", "GS", "MSI", "HM-SNV", "EBV"]
    present_labels = sorted(np.unique(np.concatenate([test_targets, test_preds])))
    present_names  = [subtype_names[i] for i in present_labels]
    print(f"\nTest Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall   : {test_metrics['recall']:.4f}")
    print(f"Test F1       : {test_metrics['f1']:.4f}")
    print(f"\nModality weights (final): {model.get_modality_weights()}")
    print(f"\nClassification Report:\n{classification_report(test_targets, test_preds, labels=present_labels, target_names=present_names, zero_division=0)}")

    # Lưu test results
    test_results = {"test_metrics": test_metrics, "modality_weights": model.get_modality_weights()}
    with open(os.path.join(args.save_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\n{'★'*60}")
    print(f"  HOÀN TẤT — Best val acc: {best_val_acc:.4f} | Test acc: {test_metrics['accuracy']:.4f}")
    print(f"{'★'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train MoXGATE model")

    # Paths
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Thư mục data_final/ chứa final_*.csv")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Thư mục lưu checkpoint và kết quả")

    # Training hyperparams
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lambda1",      type=float, default=0.01,
                        help="Hệ số regularization modality weights")
    parser.add_argument("--lambda2",      type=float, default=1e-4,
                        help="Hệ số Frobenius norm cross-attention (paper không chỉ định)")
    parser.add_argument("--patience",     type=int,   default=15,
                        help="Early stopping patience (epochs)")

    # Data
    parser.add_argument("--val_ratio",   type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--num_workers", type=int,   default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
