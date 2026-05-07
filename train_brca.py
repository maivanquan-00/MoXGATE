"""
train_brca.py
=============
Training script cho MoXGATE model — dành riêng cho BRCA (Breast Cancer).

Khác biệt so với train.py (GI):
    - Import `dataset_brca` thay vì `dataset`
    - Stratified random split (không split theo cancer type như GI)
    - Subtype names: LumA / LumB / Her2 / Basal / Normal (PAM50)
    - Test set là 10% random stratified (không phải fixed ESCA)

Hyperparameters (giữ nguyên theo paper Section 2.1.6):
    optimizer:   AdamW, lr=1e-4, weight_decay=1e-2
    batch_size:  32
    epochs:      100 (early stopping patience=15)
    focal:       γ=2, α=1 (paper Section 2.1.4)
    λ1:          0.01  (modality weight regularization)
    λ2:          1e-4  (Frobenius norm cross-attention)

Cách chạy (đường dẫn được tự động phát hiện qua config.py):
    # Chạy với đường dẫn mặc định từ config.py:
    python train_brca.py

    # Ghi đè đường dẫn nếu cần:
    python train_brca.py --data_dir /path/to/data_final_brca --save_dir /path/to/checkpoints_brca

    # Colab (chỉ cần chỉnh LOCAL_BASE / DRIVE_PROJECT_DIR trong config.py):
    !python MoXGATE/train_brca.py
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import config  # noqa: E402 — đường dẫn trung tâm, tự phát hiện Colab vs Local

# ─── Import từ dataset BRCA (KHÔNG phải dataset.py của GI) ───────────────────
from dataset_brca import build_dataloaders, BRCA_SUBTYPE_NAMES, NUM_CLASSES_BRCA
from model import MoXGATE

# ─────────────────────────────────────────────────────────────────────────────
# 0. VISUALIZATION UTILS
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Vẽ và lưu heatmap cho Confusion Matrix."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - BRCA Subtypes')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"  ✓ Saved Confusion Matrix Heatmap: {save_path}")
    except ImportError:
        print("\n  ! Skill: matplotlib/seaborn không khả dụng. Chỉ in text-based matrix.")
        print(confusion_matrix(y_true, y_pred))


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
    print("  MoXGATE — TRAINING (BRCA)")
    print(f"{'★'*60}")
    print(f"  Device     : {device}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Save dir   : {args.save_dir}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  Seed       : {args.seed}")
    print(f"  Dataset    : BRCA (PAM50 subtypes — stratified split)")
    print(f"{'★'*60}\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    # Gọi build_dataloaders của dataset_brca.py — trả về 6 giá trị (bao gồm class_weights)
    train_loader, val_loader, test_loader, scalers, dims, class_weights = build_dataloaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        val_ratio   = args.val_ratio,
        test_ratio  = args.test_ratio,
        seed        = args.seed,
        num_workers = args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MoXGATE(
        gene_dim   = dims["gene"],
        mirna_dim  = dims["mirna"],
        methyl_dim = dims["methyl"],
        num_classes = NUM_CLASSES_BRCA,  # 5 PAM50 subtypes
    ).to(device)

    # Tùy chọn: dùng class weights để compensate imbalance BRCA
    # Paper: α=1 (không weight) — bỏ comment dòng dưới nếu muốn thử
    model.set_class_weights(class_weights)

    print(f"[Train BRCA] Trainable parameters: {model.count_parameters():,}\n")

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
    best_val_loss    = float("inf")
    best_val_acc     = 0.0
    patience_counter = 0
    history          = []

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

        avg_train_loss = epoch_loss  / len(train_loader)
        avg_focal_loss = epoch_focal / len(train_loader)
        elapsed        = time.time() - t0

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
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc":         best_val_acc,
                "val_loss":        best_val_loss,
                "config":          model.config,
                "dims":            dims,
                "dataset":         "BRCA",
            }, os.path.join(args.save_dir, "best_model_brca.pt"))
            print(f"  ✓ Saved best model (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Train BRCA] Early stopping tại epoch {epoch} (patience={args.patience})")
                break

    # ── Lưu history ───────────────────────────────────────────────────────────
    history_path = os.path.join(args.save_dir, "history_brca.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Train BRCA] History đã lưu: {history_path}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  ĐÁNH GIÁ TRÊN TEST SET (BRCA — stratified random)")
    print(f"{'─'*60}")

    checkpoint = torch.load(
        os.path.join(args.save_dir, "best_model_brca.pt"),
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state"])

    test_metrics, test_preds, test_targets = evaluate(model, test_loader, device)

    # Dùng BRCA_SUBTYPE_NAMES — KHÔNG phải tên subtype GI
    present_labels = sorted(np.unique(np.concatenate([test_targets, test_preds])))
    present_names  = [BRCA_SUBTYPE_NAMES[i] for i in present_labels]

    print(f"\nTest Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall   : {test_metrics['recall']:.4f}")
    print(f"Test F1       : {test_metrics['f1']:.4f}")
    print(f"\nModality weights (final): {model.get_modality_weights()}")
    print(f"\nClassification Report:\n"
          f"{classification_report(test_targets, test_preds, labels=present_labels, target_names=present_names, zero_division=0)}")

    # ── Vẽ Confusion Matrix ──
    cm_path = os.path.join(args.save_dir, "confusion_matrix_brca.png")
    plot_confusion_matrix(test_targets, test_preds, labels=present_names, save_path=cm_path)

    # Lưu test results
    test_results = {
        "test_metrics":    test_metrics,
        "modality_weights": model.get_modality_weights(),
        "dataset":         "BRCA",
        "subtype_names":   BRCA_SUBTYPE_NAMES,
    }
    results_path = os.path.join(args.save_dir, "test_results_brca.json")
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\n{'★'*60}")
    print(f"  HOÀN TẤT BRCA — Val Acc: {best_val_acc:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | Test F1: {test_metrics['f1']:.4f}")
    print(f"{'★'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────
    return test_metrics

def parse_args():
    parser = argparse.ArgumentParser(
    parser.add_argument("--runs", type=int, default=1, help="Số lần chạy tính trung bình")
        description="Train MoXGATE model — BRCA dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Paths (default tự động từ config.py) ─────────────────────────────────
    parser.add_argument(
        "--data_dir", type=str, default=config.BRCA_FINAL_DIR,
        help="Thư mục data_final_brca/ chứa final_*.csv",
    )
    parser.add_argument(
        "--save_dir", type=str, default=config.BRCA_CHECKPOINT_DIR,
        help="Thư mục lưu checkpoint và kết quả BRCA",
    )

    # ── Training hyperparams (default từ config.py theo paper) ───────────────
    parser.add_argument("--epochs",       type=int,   default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch_size",   type=int,   default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",           type=float, default=config.DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=config.DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--lambda1",      type=float, default=config.DEFAULT_LAMBDA1,
                        help="Hệ số regularization modality weights")
    parser.add_argument("--lambda2",      type=float, default=config.DEFAULT_LAMBDA2,
                        help="Hệ số Frobenius norm cross-attention")
    parser.add_argument("--patience",     type=int,   default=config.DEFAULT_PATIENCE,
                        help="Early stopping patience (epochs)")

    # ── Data split ────────────────────────────────────────────────────────────
    parser.add_argument("--val_ratio",   type=float, default=config.DEFAULT_VAL_RATIO,
                        help="Tỉ lệ validation trên toàn bộ tập")
    parser.add_argument("--test_ratio",  type=float, default=config.DEFAULT_TEST_RATIO,
                        help="Tỉ lệ test trên toàn bộ tập")
    parser.add_argument("--seed",        type=int,   default=config.DEFAULT_SEED)
    parser.add_argument("--num_workers", type=int,   default=config.DEFAULT_NUM_WORKERS)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.runs > 1:
        import numpy as np
        metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
        base_seed = args.seed
        for i in range(args.runs):
            print(f"
{'='*50}
RUN {i+1}/{args.runs}
{'='*50}")
            args.seed = base_seed + i
            res = train(args)
            if res:
                for k in metrics:
                    if k in res: metrics[k].append(res[k])
        
        print(f"
{'='*50}
FINAL RESULTS OVER {args.runs} RUNS
{'='*50}")
        for k, v in metrics.items():
            if v:
                print(f"{k.capitalize()}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    else:
        train(args)
