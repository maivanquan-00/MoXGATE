"""
train_gat.py
==============
Training script cho mô hình HeteroMoXGATE.
Sử dụng PyTorch Geometric và tập đồ thị sinh học.
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
from dataset import build_dataloaders
from model_gat import HeteroMoXGATE

GI_SUBTYPE_NAMES = {0: "CIN", 1: "GS", 2: "MSI", 3: "HM-SNV", 4: "EBV"}
NUM_CLASSES_GI = 5

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - GI Subtypes (HeteroMoXGATE)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except ImportError:
        pass

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
        "loss":      total_loss / len(loader),
        "accuracy":  accuracy_score(all_targets, all_preds),
        "f1":        f1_score(all_targets, all_preds, average="weighted", zero_division=0),
        "precision": precision_score(all_targets, all_preds, average="weighted", zero_division=0),
        "recall":    recall_score(all_targets, all_preds, average="weighted", zero_division=0),
    }, all_preds, all_targets

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'★'*60}")
    print("  HeteroMoXGATE — TRAINING (Graph Attention + Sparsemax)")
    print(f"{'★'*60}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # DATA LOAD
    train_loader, val_loader, test_loader, scalers, dims, class_weights = build_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, 
        val_ratio=args.val_ratio, test_ratio=0.2, seed=args.seed
    )

    # GRAPH LOAD
    graph_path = os.path.join(config.BASE_DIR, "data_graph", "hetero_graph.pt")
    if not os.path.exists(graph_path):
        print("Graph not found! Please run build_graph.py first.")
        return
        
    graph_data = torch.load(graph_path)
    # Check if PyG HeteroData or simple dict
    try:
        from torch_geometric.data import HeteroData
        if isinstance(graph_data, HeteroData):
            edge_index_dict = graph_data.edge_index_dict
        else:
            edge_index_dict = graph_data["edge_index_dict"]
    except ImportError:
        edge_index_dict = graph_data["edge_index_dict"]

    # Initialize model
    model = HeteroMoXGATE(
        edge_index_dict=edge_index_dict,
        gene_dim=dims["gene"], mirna_dim=dims["mirna"], methyl_dim=dims["methyl"],
        num_classes=NUM_CLASSES_GI,
        embed_dim=256,
        heads=8
    ).to(device)

    model.set_class_weights(class_weights)
    print(f"\n[Hetero GAT] Trainable parameters: {model.count_parameters():,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    best_val_acc, patience_counter, history = 0.0, 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_focal = 0.0, 0.0
        t0 = time.time()

        for gene, mirna, methyl, labels in train_loader:
            gene, mirna, methyl, labels = gene.to(device), mirna.to(device), methyl.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, w = model(gene, mirna, methyl)
            loss, focal = model.compute_loss(logits, labels, w, lambda1=args.lambda1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_focal += focal.item()

        val_metrics, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["loss"])
        w_dict = model.get_modality_weights()

        print(f"Epoch {epoch:3d} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
              f"G={w_dict['Gene']:.2f} M={w_dict['miRNA']:.2f} C={w_dict['Methylation']:.2f} | {time.time()-t0:.1f}s")

        history.append({"epoch": epoch, "train_loss": epoch_loss/len(train_loader), **val_metrics, **w_dict})

        if val_metrics["accuracy"] >= best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "config": model.config}, 
                       os.path.join(args.save_dir, "best_model_heterogat.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience: 
                print("Early stopping!")
                break

    # -- TEST PHASE --
    # Load best checkpoint
    print("\n--- TEST EVALUATION ---")
    checkpoint = torch.load(os.path.join(args.save_dir, "best_model_heterogat.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, test_preds, test_targets = evaluate(model, test_loader, device)

    present_labels = sorted(np.unique(np.concatenate([test_targets, test_preds])))
    present_names = [GI_SUBTYPE_NAMES[i] for i in present_labels]

    print(f"\n[HeteroMoXGATE Result - GI] Accuracy: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f}")
    print(f"Classification Report:\n{classification_report(test_targets, test_preds, labels=present_labels, target_names=present_names, zero_division=0)}")
    
    cm_path = os.path.join(args.save_dir, "confusion_matrix_heterogat.png")
    plot_confusion_matrix(test_targets, test_preds, labels=present_names, save_path=cm_path)

    with open(os.path.join(args.save_dir, "history_heterogat.json"), "w") as f: 
        json.dump(history, f, indent=2)
    print(f"\nFinal Weights (Sparsemax): {model.get_modality_weights()}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=config.GI_FINAL_DIR)
    parser.add_argument("--save_dir", type=str, default=config.GI_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=16) # GAT needs lower batch size due to memory
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=config.DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--lambda1", type=float, default=config.DEFAULT_LAMBDA1)
    parser.add_argument("--patience", type=int, default=config.DEFAULT_PATIENCE)
    parser.add_argument("--val_ratio", type=float, default=config.DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    return parser.parse_args()

if __name__ == "__main__":
    train(parse_args())
