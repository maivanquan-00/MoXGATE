"""
model_sparse.py
==============
MoXGATE with Sparsemax modality fusion.
Modified from model.py to test forcing sparsity in modality weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_utils import sparsemax  # Import Sparsemax manually implemented

# ─── Reuse components from model.py ───────────────────────────────────────────
# Note: Instead of complex inheritance, we copy the structure but swap Softmax.

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=1.0, num_classes: int = 5):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        if isinstance(alpha, (list, torch.Tensor)):
            self.register_buffer('alpha', torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor([alpha] * num_classes, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        p_t   = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t = self.alpha.gather(0, targets)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        loss = -focal_weight * torch.log(p_t + 1e-8)
        return loss.mean()

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)
        self.relu        = nn.ReLU()
        self.dropout     = nn.Dropout(dropout)
        self.self_attn = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            dropout   = dropout,
            batch_first = False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.relu(self.projection(x)))
        h_seq = h.unsqueeze(1)
        z, _  = self.self_attn(h_seq, h_seq, h_seq)
        z     = z.squeeze(1)
        z = z + h
        return z

class CrossAttentionFusionSparse(nn.Module):
    """
    Sử dụng SPARSEM_AX thay cho SOFTMAX để tính trọng số modality.
    Điều này cho phép model gán hẳn trọng số 0 cho những nguồn omics không cần thiết.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 32, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.modality_logits = nn.Parameter(torch.ones(3))

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor):
        c = torch.stack([z1, z2, z3], dim=1)
        f, _ = self.cross_attn(c, c, c)
        
        # CHỖ THAY ĐỔI: Sparsemax thay vì Softmax
        w = sparsemax(self.modality_logits, dim=0)   # (3,)
        
        f_final = w[0] * f[:, 0, :] + w[1] * f[:, 1, :] + w[2] * f[:, 2, :]
        return f_final, w

class MoXGATESparse(nn.Module):
    def __init__(
        self,
        gene_dim:          int,
        mirna_dim:         int,
        methyl_dim:        int,
        embed_dim:         int   = 256,
        num_classes:       int   = 5,
        self_attn_heads:   int   = 8,
        cross_attn_heads:  int   = 32,
        dropout_encoder:   float = 0.1,
        dropout_clf:       float = 0.3,
        hidden_dim:        int   = 128,
    ):
        super().__init__()
        self.gene_encoder   = ModalityEncoder(gene_dim,   embed_dim, self_attn_heads, dropout_encoder)
        self.mirna_encoder  = ModalityEncoder(mirna_dim,  embed_dim, self_attn_heads, dropout_encoder)
        self.methyl_encoder = ModalityEncoder(methyl_dim, embed_dim, self_attn_heads, dropout_encoder)

        # Sử dụng CrossAttentionFusionSparse
        self.fusion = CrossAttentionFusionSparse(embed_dim, cross_attn_heads, dropout_encoder)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_clf),
            nn.Linear(hidden_dim, num_classes),
        )

        self.focal_loss = FocalLoss(gamma=2.0, alpha=1.0, num_classes=num_classes)
        self.config = dict(
            gene_dim=gene_dim, mirna_dim=mirna_dim, methyl_dim=methyl_dim,
            embed_dim=embed_dim, num_classes=num_classes,
            self_attn_heads=self_attn_heads, cross_attn_heads=cross_attn_heads,
            dropout_encoder=dropout_encoder, dropout_clf=dropout_clf,
            hidden_dim=hidden_dim,
            activation="sparsemax"
        )

    def set_class_weights(self, class_weights):
        self.focal_loss.alpha = torch.as_tensor(class_weights, dtype=torch.float32).to(
            self.focal_loss.alpha.device
        )

    def forward(self, x_gene, x_mirna, x_methyl):
        z_gene   = self.gene_encoder(x_gene)
        z_mirna  = self.mirna_encoder(x_mirna)
        z_methyl = self.methyl_encoder(x_methyl)
        f_final, w = self.fusion(z_gene, z_mirna, z_methyl)
        logits = self.classifier(f_final)
        return logits, w

    def compute_loss(self, logits, targets, w, lambda1=0.01, lambda2=1e-4):
        loss_focal = self.focal_loss(logits, targets)
        #loss_modality = lambda1 * torch.sum((w - 1.0 / 3) ** 2)
        loss_modality = lambda1 * torch.sum((w - 1.0) ** 2)
        w_c = self.fusion.cross_attn.in_proj_weight
        loss_frob = lambda2 * torch.norm(w_c, p='fro') ** 2
        total_loss = loss_focal + loss_modality + loss_frob
        return total_loss, loss_focal

    def get_modality_weights(self) -> dict:
        # Sparsemax thay vì Softmax
        w = sparsemax(self.fusion.modality_logits, dim=0).detach().cpu()
        return {
            'Gene':        round(w[0].item(), 4),
            'miRNA':       round(w[1].item(), 4),
            'Methylation': round(w[2].item(), 4),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Quick Test
    model = MoXGATESparse(100, 50, 200)
    x1, x2, x3 = torch.randn(2, 100), torch.randn(2, 50), torch.randn(2, 200)
    
    # Test forcing sparsity: gán 1 cái rất nhỏ
    model.fusion.modality_logits.data = torch.tensor([10.0, 1.0, -10.0])
    logits, w = model(x1, x2, x3)
    print("Modality Weights (Sparsemax):", w)
    print("Weights dict:", model.get_modality_weights())
    # Kì vọng Methylation sẽ bằng 0 nếu logits cực thấp
