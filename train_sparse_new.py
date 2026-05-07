"""
train_sparse_new.py
===================
Training script cho MoXGATE (SPARSEMAX) — Sử dụng chiến lược chia dữ liệu STRATIFIED mới cho GI.
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import config
from dataset_new import build_dataloaders_new, GI_SUBTYPE_NAMES
from model_sparse import MoXGATESparse

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - GI New Split (Sparsemax)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except ImportError: pass

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets, total_loss = [], [], 0.0
    for gene, mirna, methyl, labels in loader:
        gene, mirna, methyl, labels = gene.to(device), mirna.to(device), methyl.to(device), labels.to(device)
        logits, w = model(gene, mirna, methyl)
        loss, _ = model.compute_loss(logits, labels, w)
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())
    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(all_targets, all_preds),
        "weighted_f1": f1_score(all_targets, all_preds, average="weighted", zero_division=0),
        "macro_f1":    f1_score(all_targets, all_preds, average="macro", zero_division=0),
        "precision": precision_score(all_targets, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_targets, all_preds, average="weighted", zero_division=0),
    }, all_preds, all_targets

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Sparsemax New Split] Starting training on GI dataset...")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Dùng bộ nạp dữ liệu MỚI
    train_loader, val_loader, test_loader, scalers, dims, class_weights = build_dataloaders_new(
        data_dir=args.data_dir, batch_size=args.batch_size, 
        test_ratio=args.test_ratio, val_ratio=args.val_ratio, seed=args.seed
    )

    model = MoXGATESparse(gene_dim=dims["gene"], mirna_dim=dims["mirna"], methyl_dim=dims["methyl"]).to(device)
    model.set_class_weights(class_weights) # Bật class weights cho GI imbalanced

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    best_val_acc, patience_counter, history = 0.0, 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for gene, mirna, methyl, labels in train_loader:
            gene, mirna, methyl, labels = gene.to(device), mirna.to(device), methyl.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, w = model(gene, mirna, methyl)
            loss, _ = model.compute_loss(logits, labels, w, lambda1=args.lambda1, lambda2=args.lambda2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        val_metrics, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["loss"])
        w_dict = model.get_modality_weights()
        print(f"Epoch {epoch:3d} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Weights: G={w_dict['Gene']:.2f} M={w_dict['miRNA']:.2f} C={w_dict['Methylation']:.2f} | {time.time()-t0:.1f}s")

        history.append({"epoch": epoch, "train_loss": epoch_loss/len(train_loader), **val_metrics})

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "config": model.config}, 
                       os.path.join(args.save_dir, "best_model_sparse_gi_new.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience: break

    # Final Test
    checkpoint = torch.load(os.path.join(args.save_dir, "best_model_sparse_gi_new.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, test_preds, test_targets = evaluate(model, test_loader, device)

    present_labels = sorted(np.unique(np.concatenate([test_targets, test_preds])))
    present_names = [GI_SUBTYPE_NAMES[i] for i in present_labels]

    print(f"\n[Sparsemax New Split Result] Accuracy: {test_metrics['accuracy']:.4f} | F1: {test_metrics['weighted_f1']:.4f}")
    print(classification_report(test_targets, test_preds, target_names=present_names, zero_division=0))
    
    cm_path = os.path.join(args.save_dir, "confusion_matrix_sparse_gi_new.png")
    plot_confusion_matrix(test_targets, test_preds, labels=present_names, save_path=cm_path)
    return test_metrics




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=config.GI_FINAL_DIR)
    parser.add_argument("--save_dir", type=str, default=config.GI_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lambda1", type=float, default=0.01)
    parser.add_argument("--lambda2", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1, help="Số lần chạy lấy trung bình trên Colab")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.runs > 1:
        import numpy as np
        metrics = {'accuracy': [], 'weighted_f1': [], 'macro_f1': [], 'precision': [], 'recall': []}
        base_seed = args.seed
        for i in range(args.runs):
            print(f"\n{'='*50}\nRUN {i+1}/{args.runs}\n{'='*50}")
            args.seed = base_seed + i
            res = train(args)
            if res:
                for k in metrics:
                    if k in res: metrics[k].append(res[k])
        
        print(f"\n{'='*50}\nFINAL RESULTS OVER {args.runs} RUNS\n{'='*50}")
        for k, v in metrics.items():
            if v:
                name_map = {"accuracy": "Accuracy", "weighted_f1": "Weighted F1", "macro_f1": "Macro F1", "precision": "Precision", "recall": "Recall"}
                print(f"{name_map.get(k, k.capitalize())}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    else:
        train(args)
