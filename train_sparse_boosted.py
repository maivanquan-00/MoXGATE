"""
train_sparse_boosted.py
========================
Thử nghiệm cải thiện hiệu suất cho lớp HM-SNV trong Sparsemax bằng 2 kỹ thuật:

  Giải pháp 1 — Boosted Class Weight:
      Ghi đè trọng số lớp HM-SNV lên 30x thay vì ~8.5x được tính tự động.
      Điều này ép Focal Loss "chú ý" đặc biệt đến mỗi mẫu HM-SNV sai.

  Giải pháp 2 — Custom Prediction Threshold:
      Thay vì dùng argmax (ngưỡng ngầm định 50%), dùng ngưỡng riêng 
      cho từng lớp. HM-SNV được gán ngưỡng thấp hơn (~0.15) để 
      model "dám" dự đoán lớp này dù xác suất nhỏ.

Script sẽ báo cáo CẢ HAI phiên bản kết quả:
  - Standard (argmax) → để so sánh công bằng với các file trước
  - Threshold-adjusted → để đánh giá hiệu quả của Giải pháp 2

Chạy:
    !python MoXGATE/train_sparse_boosted.py
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report, confusion_matrix)

import config
from dataset_new import build_dataloaders_new, GI_SUBTYPE_NAMES
from model_sparse import MoXGATESparse

# ─────────────────────────────────────────────────────────────────────────────
# GIẢI PHÁP 1: Boosted class weights cho HM-SNV
# ─────────────────────────────────────────────────────────────────────────────
# Trọng số tự động tính ra ~8.5x cho HM-SNV. Ta ghi đè thủ công.
# Các lớp khác giữ nguyên trọng số tự động.
HM_SNV_BOOST = 30.0   # Tăng lên 30x — thử nghiệm

# ─────────────────────────────────────────────────────────────────────────────
# GIẢI PHÁP 2: Ngưỡng dự đoán riêng cho từng lớp
# ─────────────────────────────────────────────────────────────────────────────
# Với argmax: một lớp thắng khi xác suất > 50% (ngưỡng ngầm định).
# Ở đây chúng ta hạ ngưỡng của HM-SNV xuống để dễ được chọn hơn.
# Chú ý: ngưỡng thấp hơn sẽ tăng Recall nhưng có thể giảm Precision.
THRESHOLDS = {
    0: 0.50,   # CIN    — giữ nguyên
    1: 0.50,   # GS     — giữ nguyên
    2: 0.50,   # MSI    — giữ nguyên
    3: 0.15,   # HM-SNV — hạ xuống 0.15
    4: 0.40,   # EBV    — hơi hạ một chút
}


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, labels, save_path, title=""):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                    xticklabels=labels, yticklabels=labels)
        plt.title(title or 'Confusion Matrix (Sparsemax Boosted)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    except ImportError:
        print("  ! matplotlib / seaborn not available.")


# ─────────────────────────────────────────────────────────────────────────────
# GIẢI PHÁP 2: Hàm dự đoán có ngưỡng riêng từng lớp
# ─────────────────────────────────────────────────────────────────────────────

def predict_with_thresholds(probs: np.ndarray, thresholds: dict) -> np.ndarray:
    """
    Thay vì argmax, dự đoán nhãn dựa trên ngưỡng riêng mỗi lớp.

    Cơ chế:
        1. Với mỗi mẫu, kiểm tra xem lớp nào có xác suất vượt ngưỡng của nó.
        2. Nếu có nhiều lớp vượt ngưỡng, chọn lớp có xác suất CAO NHẤT.
        3. Nếu không lớp nào vượt ngưỡng → fallback về argmax thông thường.

    Args:
        probs:      (N, num_classes) - xác suất sau softmax
        thresholds: dict {class_idx: threshold_value}

    Returns:
        preds: (N,) numpy array nhãn dự đoán
    """
    n_samples, n_classes = probs.shape
    threshold_arr = np.array([thresholds.get(c, 0.5) for c in range(n_classes)])  # (C,)
    
    # Mask: True nếu xác suất vượt ngưỡng của lớp đó
    above_threshold = probs > threshold_arr[None, :]  # (N, C)
    
    preds = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        candidates = np.where(above_threshold[i])[0]
        if len(candidates) > 0:
            # Nhiều lớp vượt ngưỡng → chọn lớp có xác suất cao nhất trong candidates
            preds[i] = candidates[np.argmax(probs[i, candidates])]
        else:
            # Không lớp nào vượt ngưỡng → argmax thông thường
            preds[i] = np.argmax(probs[i])
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION — trả về cả probs để dùng cho threshold prediction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_preds, all_targets, total_loss = [], [], [], 0.0

    for gene, mirna, methyl, labels in loader:
        gene, mirna, methyl, labels = (gene.to(device), mirna.to(device),
                                       methyl.to(device), labels.to(device))
        logits, w = model(gene, mirna, methyl)
        loss, _ = model.compute_loss(logits, labels, w)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=-1).cpu().numpy()   # (B, C)
        preds = np.argmax(probs, axis=-1)                      # argmax
        all_probs.append(probs)
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    all_probs   = np.vstack(all_probs)          # (N, C)
    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    metrics = {
        "loss":      total_loss / len(loader),
        "accuracy":  accuracy_score(all_targets, all_preds),
        "f1":        f1_score(all_targets, all_preds, average="weighted", zero_division=0),
        "precision": precision_score(all_targets, all_preds, average="weighted", zero_division=0),
        "recall":    recall_score(all_targets, all_preds, average="weighted", zero_division=0),
    }
    return metrics, all_probs, all_preds, all_targets


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'★'*60}")
    print("  MoXGATE — SPARSEMAX BOOSTED (GI New Split)")
    print(f"  HM-SNV boost: {HM_SNV_BOOST}x | HM-SNV threshold: {THRESHOLDS[3]}")
    print(f"{'★'*60}\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Data ──
    train_loader, val_loader, test_loader, scalers, dims, class_weights = build_dataloaders_new(
        data_dir=args.data_dir, batch_size=args.batch_size,
        test_ratio=args.test_ratio, val_ratio=args.val_ratio, seed=args.seed
    )

    # ── Giải pháp 1: Ghi đè class weight HM-SNV ──
    # class_weights[3] là HM-SNV (index 3 = HM-SNV)
    original_hmsnv_weight = class_weights[3]
    class_weights[3] = HM_SNV_BOOST
    print(f"[Boost] HM-SNV weight: {original_hmsnv_weight:.2f}x → {HM_SNV_BOOST}x")
    print(f"[Boost] Tất cả weights: {dict(zip(GI_SUBTYPE_NAMES.values(), class_weights.tolist()))}\n")

    # ── Model ──
    model = MoXGATESparse(
        gene_dim=dims["gene"], mirna_dim=dims["mirna"], methyl_dim=dims["methyl"]
    ).to(device)
    model.set_class_weights(class_weights)

    print(f"[Model] Trainable parameters: {model.count_parameters():,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    best_val_acc, patience_counter = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for gene, mirna, methyl, labels in train_loader:
            gene, mirna, methyl, labels = (gene.to(device), mirna.to(device),
                                           methyl.to(device), labels.to(device))
            optimizer.zero_grad()
            logits, w = model(gene, mirna, methyl)
            loss, _ = model.compute_loss(logits, labels, w,
                                         lambda1=args.lambda1, lambda2=args.lambda2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["loss"])
        w_dict = model.get_modality_weights()

        print(f"Epoch {epoch:3d} | Loss: {epoch_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
              f"G={w_dict['Gene']:.2f} M={w_dict['miRNA']:.2f} C={w_dict['Methylation']:.2f} | "
              f"{time.time()-t0:.1f}s")

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "config": model.config},
                       os.path.join(args.save_dir, "best_model_sparse_boosted.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stop] Epoch {epoch} — patience={args.patience}")
                break

    # ── Test Evaluation ──
    checkpoint = torch.load(os.path.join(args.save_dir, "best_model_sparse_boosted.pt"),
                            map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, test_probs, test_preds_argmax, test_targets = evaluate(model, test_loader, device)

    present_labels = sorted(np.unique(np.concatenate([test_targets, test_preds_argmax])))
    present_names  = [GI_SUBTYPE_NAMES[i] for i in present_labels]

    # ── Kết quả 1: Standard argmax (so sánh fair) ──
    print(f"\n{'─'*60}")
    print("  KẾT QUẢ 1 — Standard argmax (Boosted Weight Only)")
    print(f"{'─'*60}")
    print(f"  Accuracy : {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f}")
    print(classification_report(test_targets, test_preds_argmax,
                                labels=present_labels, target_names=present_names,
                                zero_division=0))

    plot_confusion_matrix(test_targets, test_preds_argmax, present_names,
                          os.path.join(args.save_dir, "cm_sparse_boosted_argmax.png"),
                          title="Sparsemax Boosted — argmax")

    # ── Kết quả 2: Threshold-adjusted ──
    test_preds_thresh = predict_with_thresholds(test_probs, THRESHOLDS)
    present_labels_t  = sorted(np.unique(np.concatenate([test_targets, test_preds_thresh])))
    present_names_t   = [GI_SUBTYPE_NAMES[i] for i in present_labels_t]

    acc_t  = accuracy_score(test_targets, test_preds_thresh)
    f1_t   = f1_score(test_targets, test_preds_thresh, average="weighted", zero_division=0)

    print(f"\n{'─'*60}")
    print(f"  KẾT QUẢ 2 — Threshold-adjusted (HM-SNV threshold={THRESHOLDS[3]})")
    print(f"{'─'*60}")
    print(f"  Accuracy : {acc_t:.4f} | F1: {f1_t:.4f}")
    print(classification_report(test_targets, test_preds_thresh,
                                labels=present_labels_t, target_names=present_names_t,
                                zero_division=0))

    plot_confusion_matrix(test_targets, test_preds_thresh, present_names_t,
                          os.path.join(args.save_dir, "cm_sparse_boosted_threshold.png"),
                          title=f"Sparsemax Boosted — threshold HM-SNV={THRESHOLDS[3]}")

    # ── In xác suất dự đoán cho các mẫu HM-SNV trong test ──
    hmsnv_idx = np.where(test_targets == 3)[0]
    print(f"\n{'─'*60}")
    print(f"  Debug: {len(hmsnv_idx)} mẫu HM-SNV trong test set")
    print(f"  (Xác suất dự đoán để hiểu tại sao model nhầm)")
    print(f"{'─'*60}")
    for i in hmsnv_idx:
        prob_str = " | ".join([f"{GI_SUBTYPE_NAMES[c]}={test_probs[i,c]:.3f}" for c in range(5)])
        pred_argmax = GI_SUBTYPE_NAMES[test_preds_argmax[i]]
        pred_thresh = GI_SUBTYPE_NAMES[test_preds_thresh[i]]
        print(f"  Sample {i}: [{prob_str}]")
        print(f"    → argmax: {pred_argmax} | threshold: {pred_thresh} | true: HM-SNV")

    print(f"\n{'★'*60}")
    print(f"  Final Modality Weights: {model.get_modality_weights()}")
    print(f"{'★'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Sparsemax Boosted — GI New Split")
    parser.add_argument("--data_dir",     type=str,   default=config.GI_FINAL_DIR)
    parser.add_argument("--save_dir",     type=str,   default=config.GI_CHECKPOINT_DIR)
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lambda1",      type=float, default=0.01)
    parser.add_argument("--lambda2",      type=float, default=1e-4)
    parser.add_argument("--patience",     type=int,   default=15)
    parser.add_argument("--test_ratio",   type=float, default=0.2)
    parser.add_argument("--val_ratio",    type=float, default=0.1)
    parser.add_argument("--seed",         type=int,   default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
