import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import HeteroConv, GATConv
except ImportError:
    pass

from sparse_utils import sparsemax
from model import FocalLoss

class HeteroMoXGATE(nn.Module):
    def __init__(
        self,
        edge_index_dict,
        gene_dim, mirna_dim, methyl_dim,
        embed_dim=256,
        num_classes=5,
        heads=8,
        dropout_encoder=0.1,
        dropout_clf=0.3,
        hidden_dim=128
    ):
        """
        MoXGATE using Heterogeneous Graph Attention Networks.
        
        Args:
            edge_index_dict: Dictionary chứa cấu trúc đồ thị từ data_graph/hetero_graph.pt
        """
        super().__init__()
        self.edge_index_dict = edge_index_dict
        
        # In tabular data, each node (gene/mirna/cpg) has 1 scalar feature (expression/beta value)
        self.gene_proj = nn.Linear(1, embed_dim)
        self.mirna_proj = nn.Linear(1, embed_dim)
        self.methyl_proj = nn.Linear(1, embed_dim)
        
        # Hetero GAT: Xử lý mỗi modality interacting với nhau qua message passing
        self.conv = HeteroConv({
            ('gene', 'interacts', 'gene'): GATConv(embed_dim, embed_dim // heads, heads=heads, add_self_loops=False),
            ('mirna', 'targets', 'gene'): GATConv((-1, -1), embed_dim // heads, heads=heads, add_self_loops=False),
            ('cpg', 'regulates', 'gene'): GATConv((-1, -1), embed_dim // heads, heads=heads, add_self_loops=False),
            ('gene', 'rev_targets', 'mirna'): GATConv((-1, -1), embed_dim // heads, heads=heads, add_self_loops=False),
            ('gene', 'rev_regulates', 'cpg'): GATConv((-1, -1), embed_dim // heads, heads=heads, add_self_loops=False),
            ('gene', 'rev_interacts', 'gene'): GATConv(embed_dim, embed_dim // heads, heads=heads, add_self_loops=False),
        }, aggr='sum')
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_encoder)

        self.modality_logits = nn.Parameter(torch.ones(3))

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_clf),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.focal_loss = FocalLoss(gamma=2.0, alpha=1.0, num_classes=num_classes)
        
        self.config = dict(
            gene_dim=gene_dim, mirna_dim=mirna_dim, methyl_dim=methyl_dim,
            embed_dim=embed_dim, num_classes=num_classes,
            heads=heads, dropout_encoder=dropout_encoder, dropout_clf=dropout_clf,
            hidden_dim=hidden_dim,
            activation="sparsemax_gat"
        )

    def set_class_weights(self, class_weights):
        self.focal_loss.alpha = torch.as_tensor(class_weights, dtype=torch.float32).to(self.focal_loss.alpha.device)

    def forward(self, x_gene, x_mirna, x_methyl):
        """
        x_gene: (B, num_genes)
        x_mirna: (B, num_mirnas)
        x_methyl: (B, num_cpgs)
        """
        B = x_gene.size(0)
        device = x_gene.device
        
        n_gene = x_gene.size(1)
        n_mirna = x_mirna.size(1)
        n_methyl = x_methyl.size(1)
        
        # Batching Edge Index: Tự động nhân bản đồ thị B lần để xử lý song song (B * N)
        if getattr(self, '_cached_B', None) != B:
            batched_edge_index = {}
            for edge_type, edge_index in self.edge_index_dict.items():
                src_type, rel, dst_type = edge_type
                n_src = n_gene if src_type == 'gene' else (n_mirna if src_type == 'mirna' else n_methyl)
                n_dst = n_gene if dst_type == 'gene' else (n_mirna if dst_type == 'mirna' else n_methyl)
                
                E = edge_index.size(1)
                
                # Tạo offset cho batch
                src_offset = torch.arange(B, device=device) * n_src
                dst_offset = torch.arange(B, device=device) * n_dst
                
                src_offset = src_offset.view(B, 1).expand(B, E).contiguous().view(-1)
                dst_offset = dst_offset.view(B, 1).expand(B, E).contiguous().view(-1)
                
                batched_src = edge_index[0].to(device).repeat(B) + src_offset
                batched_dst = edge_index[1].to(device).repeat(B) + dst_offset
                batched_edge_index[edge_type] = torch.stack([batched_src, batched_dst], dim=0)
                
            self._cached_edge_index_dict = batched_edge_index
            self._cached_B = B
            
        edge_index_dict = self._cached_edge_index_dict

        # Chuyển Feature -> (B*N, 1) và chiếu lên Embed Dim
        x_g = self.relu(self.gene_proj(x_gene.view(-1, 1)))
        x_m = self.relu(self.mirna_proj(x_mirna.view(-1, 1)))
        x_c = self.relu(self.methyl_proj(x_methyl.view(-1, 1)))
        
        x_g = self.dropout(x_g)
        x_m = self.dropout(x_m)
        x_c = self.dropout(x_c)

        x_dict = {'gene': x_g, 'mirna': x_m, 'cpg': x_c}
        
        # GATConv Message Passing
        x_dict = self.conv(x_dict, edge_index_dict)
        
        # Readout: Group by patient (Global Mean Pool)
        # Reshape lại thành (B, N, embed) và average pooling
        z_gene = x_dict['gene'].view(B, n_gene, -1).mean(dim=1)
        z_mirna = x_dict['mirna'].view(B, n_mirna, -1).mean(dim=1)
        z_methyl = x_dict['cpg'].view(B, n_methyl, -1).mean(dim=1)
        
        # Sparsemax fusion (Modality Attention)
        w = sparsemax(self.modality_logits, dim=0)
        f_final = w[0] * z_gene + w[1] * z_mirna + w[2] * z_methyl
        
        logits = self.classifier(f_final)
        return logits, w

    def compute_loss(self, logits, targets, w, lambda1=0.01, lambda2=0):
        loss_focal = self.focal_loss(logits, targets)
        loss_modality = lambda1 * torch.sum((w - 1.0) ** 2)
        total_loss = loss_focal + loss_modality
        return total_loss, loss_focal
        
    def get_modality_weights(self) -> dict:
        w = sparsemax(self.modality_logits, dim=0).detach().cpu()
        return {
            'Gene':        round(w[0].item(), 4),
            'miRNA':       round(w[1].item(), 4),
            'Methylation': round(w[2].item(), 4),
        }
        
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    pass
