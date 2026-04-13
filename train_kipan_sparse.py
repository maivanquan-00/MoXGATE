"""
train_kipan_sparse.py
=====================
Training script cho MoXGATE (SPARSEMAX) — KIPAN (Kidney Cancer).

3 phân lớp: KICH (0) / KIRC (1) / KIRP (2)

Cách chạy:
    !python MoXGATE/train_kipan_sparse.py
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
from dataset_kipan import build_dataloaders, KIPAN_SUBTYPE_NAMES, NUM_CLASSES_KIPAN
from model_sparse import MoXGATESparse


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - Kidney Cancer KIPAN (Sparsemax)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    except ImportError:
        pass


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets, total_loss = [], [], 0.0
    for gene, mirna, methyl, labels in loader:
        gene, mirna, methyl, labels = (gene.to(device), mirna.to(device),
                                       methyl.to(device), labels.to(device))
        logits, w = model(gene, mirna, methyl)
        loss, _ = model.compute_loss(logits, labels, w)
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())
    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    return {
        "loss":      total_loss / len(loader),
        "accuracy":  accuracy_score(all_targets, all_preds),
        "f1":        f1_score(all_targets, all_preds, average="weighted", zero_division=0),
        "precision": precision_score(all_targets, all_preds, average="weighted", zero_division=0),
        "recall":    recall_score(all_targets, all_preds, average="weighted", zero_division=0),
    }, all_preds, all_targets


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'★'*60}")
    print("  MoXGATE — TRAINING (KIPAN Kidney Cancer — Sparsemax)")
    print(f"{'★'*60}\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, test_loader, scalers, dims, class_weights = build_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        seed=args.seed, num_workers=args.num_workers,
    )

    model = MoXGATESparse(
        gene_dim=dims["gene"], mirna_dim=dims["mirna"], methyl_dim=dims["methyl"],
        num_classes=NUM_CLASSES_KIPAN,
    ).to(device)
    model.set_class_weights(class_weights)

    print(f"[Train KIPAN Sparse] Trainable parameters: {model.count_parameters():,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    best_val_acc, patience_counter, history = 0.0, 0, []

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

        val_metrics, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["loss"])
        w_dict = model.get_modality_weights()

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
              f"G={w_dict['Gene']:.2f} M={w_dict['miRNA']:.2f} C={w_dict['Methylation']:.2f} | "
              f"{time.time()-t0:.1f}s")

        history.append({"epoch": epoch, "train_loss": epoch_loss/len(train_loader),
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                        **{f"w_{k}": v for k, v in w_dict.items()}})

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "config": model.config},
                       os.path.join(args.save_dir, "best_model_kipan_sparse.pt"))
            print(f"  ✓ Saved best model (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Train KIPAN Sparse] Early stopping tại epoch {epoch}")
                break

    # ── Test ──
    checkpoint = torch.load(os.path.join(args.save_dir, "best_model_kipan_sparse.pt"),
                            map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, test_preds, test_targets = evaluate(model, test_loader, device)

    present_labels = sorted(np.unique(np.concatenate([test_targets, test_preds])))
    present_names  = [KIPAN_SUBTYPE_NAMES[i] for i in present_labels]

    print(f"\n{'─'*60}")
    print("  ĐÁNH GIÁ TEST SET (KIPAN Sparsemax)")
    print(f"{'─'*60}")
    print(f"\nTest Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Test F1       : {test_metrics['f1']:.4f}")
    print(f"\nModality weights (Sparsemax): {model.get_modality_weights()}")
    print(f"\n{classification_report(test_targets, test_preds, labels=present_labels, target_names=present_names, zero_division=0)}")

    plot_confusion_matrix(test_targets, test_preds, present_names,
                          os.path.join(args.save_dir, "confusion_matrix_kipan_sparse.png"))

    with open(os.path.join(args.save_dir, "history_kipan_sparse.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'★'*60}")
    print(f"  HOÀN TẤT KIPAN Sparse — Val Acc: {best_val_acc:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | Test F1: {test_metrics['f1']:.4f}")
    print(f"{'★'*60}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MoXGATE Sparse — KIPAN")
    parser.add_argument("--data_dir",     type=str,   default=config.KIPAN_FINAL_DIR)
    parser.add_argument("--save_dir",     type=str,   default=config.KIPAN_CHECKPOINT_DIR)
    parser.add_argument("--epochs",       type=int,   default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch_size",   type=int,   default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",           type=float, default=config.DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=config.DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--lambda1",      type=float, default=config.DEFAULT_LAMBDA1)
    parser.add_argument("--lambda2",      type=float, default=config.DEFAULT_LAMBDA2)
    parser.add_argument("--patience",     type=int,   default=config.DEFAULT_PATIENCE)
    parser.add_argument("--val_ratio",    type=float, default=config.DEFAULT_VAL_RATIO)
    parser.add_argument("--test_ratio",   type=float, default=config.DEFAULT_TEST_RATIO)
    parser.add_argument("--seed",         type=int,   default=config.DEFAULT_SEED)
    parser.add_argument("--num_workers",  type=int,   default=config.DEFAULT_NUM_WORKERS)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
