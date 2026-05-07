import os
import argparse
import torch
import numpy as np
import pandas as pd

try:
    from captum.attr import IntegratedGradients
except ImportError:
    print("WARNING: Cần cài đặt dict Captum (pip install captum) để chạy feature importance.")
    IntegratedGradients = None

try:
    import gseapy as gp
except ImportError:
    print("WARNING: Cần cài đặt gseapy (pip install gseapy) để chạy GSEA.")
    gp = None

import config
from model_gat import HeteroMoXGATE
from dataset import build_dataloaders

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x_gene, x_mirna, x_methyl):
        # Chỉ trả về logits cho Captum
        logits, _ = self.model(x_gene, x_mirna, x_methyl)
        return logits

def interpret(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=== Bước 1: Load Dữ liệu và Mô hình ===")
    
    # Không cần chia ngẫu nhiên lại từ đầu vì ta dùng seed cố định
    _, _, test_loader, _, dims, _ = build_dataloaders(
        data_dir=config.GI_FINAL_DIR,
        batch_size=args.batch_size, 
        val_ratio=config.DEFAULT_VAL_RATIO, 
        test_ratio=0.2, 
        seed=config.DEFAULT_SEED
    )
    
    graph_path = os.path.join(config.BASE_DIR, "data_graph", "hetero_graph.pt")
    if not os.path.exists(graph_path):
        print("Graph not found! Run build_graph.py first.")
        return
        
    graph_data = torch.load(graph_path)
    try:
        from torch_geometric.data import HeteroData
        if isinstance(graph_data, HeteroData):
            edge_index_dict = graph_data.edge_index_dict
        else:
            edge_index_dict = graph_data["edge_index_dict"]
    except ImportError:
        edge_index_dict = graph_data["edge_index_dict"]
        
    model = HeteroMoXGATE(
        edge_index_dict=edge_index_dict,
        gene_dim=dims["gene"], mirna_dim=dims["mirna"], methyl_dim=dims["methyl"],
        num_classes=5, embed_dim=256, heads=8
    ).to(device)
    
    model_path = os.path.join(config.GI_CHECKPOINT_DIR, "best_model_heterogat.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device)['model_state'])
    else:
        print("Warning: Không tìm thấy checkpoint đã train. Dùng random weights.")
    
    model.eval()
    wrapped_model = ModelWrapper(model).to(device)
    
    if IntegratedGradients is None:
        return
        
    ig = IntegratedGradients(wrapped_model)
    
    print("\n=== Bước 2: Tính Feature Importance trên Test Set ===")
    
    # Lấy 1 batch từ test loader
    gene, mirna, methyl, labels = next(iter(test_loader))
    gene = gene.to(device)
    mirna = mirna.to(device)
    methyl = methyl.to(device)
    labels = labels.to(device)
    
    # Phân loại để xem bệnh nhân
    with torch.no_grad():
        logits = wrapped_model(gene, mirna, methyl)
        preds = logits.argmax(dim=1)
    
    # Chọn ngẫu nhiên 2 bệnh nhân có chuẩn đoán là EBV (class 4) hoặc MSI (class 2) nếu có
    targets_to_explain = []
    for i in range(len(labels)):
        if labels[i].item() in [2, 4]:
            targets_to_explain.append((i, labels[i].item()))
            if len(targets_to_explain) == 2:
                break
                
    if not targets_to_explain:
        targets_to_explain = [(0, labels[0].item()), (1, labels[1].item())]
        
    print(f"Chọn Bệnh nhân Index: {[t[0] for t in targets_to_explain]} (Classes: {[t[1] for t in targets_to_explain]})")
    
    for pat_idx, true_label in targets_to_explain:
        print(f"\n--- Giải thích cho bệnh nhân {pat_idx} (Label {true_label}) ---")
        g_in = gene[pat_idx:pat_idx+1]
        m_in = mirna[pat_idx:pat_idx+1]
        c_in = methyl[pat_idx:pat_idx+1]
        
        attributions, delta = ig.attribute(
            inputs=(g_in, m_in, c_in),
            target=true_label,
            return_convergence_delta=True
        )
        
        attr_gene = attributions[0].squeeze().cpu().detach().numpy()
        attr_cpg = attributions[2].squeeze().cpu().detach().numpy()
        
        # Read column names if available
        gene_names = []
        try:
            with open(os.path.join(config.GI_FINAL_DIR, "final_gene_symbol.csv"), "r") as f:
                gene_names = f.readline().strip().split(',')[1:]
        except:
            gene_names = [f"Gene_{i}" for i in range(len(attr_gene))]
            
        top_gene_indices = np.argsort(np.abs(attr_gene))[-50:][::-1]
        top_genes = [gene_names[i].split('.')[0] for i in top_gene_indices]
        
        print(f"Top 10 Genes ảnh hưởng lớn nhất:")
        for i in range(10):
            idx = top_gene_indices[i]
            print(f"  {gene_names[idx]}: {attr_gene[idx]:.4f}")
            
        if gp is not None:
            print(f">>> Chạy GSEA (Pathway Enrichment) cho tập Top 50 Genes của BN {pat_idx}...")
            try:
                enr = gp.enrichr(gene_list=top_genes,
                                 gene_sets=['KEGG_2021_Human', 'Reactome_2022'],
                                 organism='human', 
                                 outdir=f'enrichr_results_pat_{pat_idx}')
                
                print("Top Pathways:")
                print(enr.results[['Term', 'Adjusted P-value', 'Genes']].head(3))
            except Exception as e:
                print("Lỗi khi chạy gseapy:", e)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()

if __name__ == "__main__":
    interpret(parse_args())
