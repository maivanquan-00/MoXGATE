"""
model.py
========
MoXGATE: Modality-Aware Cross-Attention for Multi-Omic Cancer Subtype Classification.
Reimplemented from: Dip et al., AI4NA workshop at ICLR 2025.

Architecture (3 components theo Figure 1):

    (a) Modality-Specific Self-Attention Encoding
        ─ Mỗi omics được encode độc lập qua Linear + ReLU + MultiheadSelfAttention
        ─ Self-attention là cross-sample attention: mỗi bệnh nhân trong batch là 1 "token"
        ─ 8 heads, dropout 0.1

    (b) Modality-Weighted Cross-Attention Fusion
        ─ Stack 3 modality outputs thành sequence 3 tokens
        ─ Cross-attention: học tương tác giữa 3 nguồn omics
        ─ Learnable weights w = softmax([w1,w2,w3]) → F_final = Σ wi * Fi
        ─ 32 heads, embed_dim = 256

    (c) Subtype Classification
        ─ FC(256→128) → ReLU → Dropout(0.3) → FC(128→5)
        ─ Focal Loss (γ=2, α=1) để xử lý class imbalance

Hàm mất mát tổng quát:
    L = L_focal + λ1‖w - 1/3‖² + λ2‖W_c‖²_F

Hyperparameters (theo paper, Section 2.1.6):
    embed_dim        = 256
    self_attn_heads  = 8,   dropout = 0.1
    cross_attn_heads = 32
    classifier dropout = 0.3
    hidden_dim       = 128
    optimizer: AdamW, lr=1e-4, weight_decay=1e-2
    focal: γ=2, α=1

Input dimensions (từ GDC Harmonized data):
    gene_dim   ≈ 19,962  (STAR-TPM, protein-coding, GENCODE v36)
    mirna_dim  ≈  1,881  (miRBase v22 stem-loop)
    methyl_dim ≈ 23,111  (27k+450k, manifest filter)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. FOCAL LOSS
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al. 2017) cho class imbalance.

    L = -Σ α_i (1 - p_i)^γ · y_i · log(p_i)

    Paper settings: γ=2, α=1 (đồng đều tất cả class).

    Args:
        gamma:       Focusing parameter — giảm trọng số lớp dễ phân loại (default 2)
        alpha:       Class weighting (default 1 = không ưu tiên lớp nào)
        num_classes: Số lớp (5)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, num_classes: int = 5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, num_classes) — raw model output (trước softmax)
            targets: (B,)             — nhãn class (0-indexed)
        Returns:
            Scalar loss
        """
        probs = F.softmax(logits, dim=-1)                              # (B, C)
        p_t   = probs.gather(1, targets.unsqueeze(1)).squeeze(1)       # (B,)
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma          # (B,)
        loss = -focal_weight * torch.log(p_t + 1e-8)                   # (B,)
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODALITY-SPECIFIC SELF-ATTENTION ENCODER  (Bước a)
# ─────────────────────────────────────────────────────────────────────────────

class ModalityEncoder(nn.Module):
    """
    Encode 1 omics modality qua Linear projection + Self-Attention.

    Theo Eq. (1)–(5) trong paper:
        H_m = ReLU(X_m W_m + b_m)          — (B, embed_dim)
        Q, K, V = H_m W_Q, H_m W_K, H_m W_V
        A_m = softmax(QK^T / √d)
        Z_m = A_m V_m + H_m                 — residual connection

    Self-attention là cross-sample attention:
        Mỗi bệnh nhân trong batch là 1 "token" (seq_len = batch_size, batch_size_attn = 1).
        Model học tương quan giữa các bệnh nhân trong cùng batch.

    Args:
        input_dim:  Số features đầu vào của omics (vd: 19962 cho Gene)
        embed_dim:  Chiều sau projection (256)
        num_heads:  Số attention heads (8)
        dropout:    Dropout rate (0.1)
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)
        self.relu        = nn.ReLU()
        self.dropout     = nn.Dropout(dropout)

        # batch_first=False → expect (seq_len, batch_size_attn, embed_dim)
        # Ta sẽ dùng: seq_len=B (số sample), batch_size_attn=1
        self.self_attn = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            dropout   = dropout,
            batch_first = False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            z: (B, embed_dim)
        """
        # Bước 2.1: Linear projection + ReLU
        h = self.dropout(self.relu(self.projection(x)))  # (B, embed_dim)

        # Bước 2.2: Self-attention (cross-sample)
        # h.unsqueeze(1) → (B, 1, embed_dim)
        # Với batch_first=False: dim0=seq_len=B, dim1=batch=1, dim2=embed
        h_seq = h.unsqueeze(1)                           # (B, 1, embed_dim)
        z, _  = self.self_attn(h_seq, h_seq, h_seq)     # (B, 1, embed_dim)
        z     = z.squeeze(1)                             # (B, embed_dim)

        # Bước 2.3: Residual connection (Eq. 5)
        z = z + h                                        # (B, embed_dim)
        return z


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODALITY-WEIGHTED CROSS-ATTENTION FUSION  (Bước b)
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Fuse 3 omics modalities qua Cross-Attention + Learnable Modality Weights.

    Theo Eq. (6)–(11) trong paper:
        C = [Z1, Z2, Z3]  ∈ R^{B×3×d}     — 3 modalities = 3 tokens
        Q_c, K_c, V_c = C W_Q^c, C W_K^c, C W_V^c
        A_c = softmax(Q_c K_c^T / √d)
        F = A_c V_c                          — (B, 3, d)
        w = softmax([w1,w2,w3])              — Σwi = 1
        F_final = w1*F[:,0] + w2*F[:,1] + w3*F[:,2]   — (B, d)

    Args:
        embed_dim:  Chiều embedding (256)
        num_heads:  Số cross-attention heads (32)
        dropout:    Dropout rate
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 32,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,   # (B, seq=3, embed)
        )

        # Learnable raw logits → softmax → modality weights
        # Khởi tạo đều: w1 = w2 = w3 = 1/3
        self.modality_logits = nn.Parameter(torch.ones(3))

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
    ):
        """
        Args:
            z1: (B, embed_dim) — output của gene encoder
            z2: (B, embed_dim) — output của mirna encoder
            z3: (B, embed_dim) — output của methyl encoder
        Returns:
            f_final: (B, embed_dim) — fused representation
            w:       (3,)           — modality weights (sau softmax)
        """
        # Stack 3 modalities thành sequence 3 "tokens": (B, 3, embed_dim)
        c = torch.stack([z1, z2, z3], dim=1)         # (B, 3, embed_dim)

        # Cross-attention qua 3 modality tokens
        f, _ = self.cross_attn(c, c, c)              # (B, 3, embed_dim)

        # Modality weights: softmax đảm bảo Σwi = 1 (Eq. 10)
        w = F.softmax(self.modality_logits, dim=0)   # (3,)

        # Weighted sum (Eq. 11): F_final = w1*F1 + w2*F2 + w3*F3
        f_final = w[0] * f[:, 0, :] + w[1] * f[:, 1, :] + w[2] * f[:, 2, :]

        return f_final, w


# ─────────────────────────────────────────────────────────────────────────────
# 4. MOXGATE — FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class MoXGATE(nn.Module):
    """
    MoXGATE: Modality-Aware Cross-Attention for Multi-Omic Cancer Subtype Classification.

    Full pipeline:
        x_gene, x_mirna, x_methyl
            ↓ (a)  3 × ModalityEncoder (self-attention, 8 heads)
        z_gene, z_mirna, z_methyl
            ↓ (b)  CrossAttentionFusion (32 heads) + modality weights
        f_final
            ↓ (c)  FC(256→128) → ReLU → Dropout(0.3) → FC(128→5)
        logits

    Loss:
        L = L_focal + λ1‖w - 1/3‖² + λ2‖W_c‖²_F

    Args:
        gene_dim:        Số gene features
        mirna_dim:       Số miRNA features
        methyl_dim:      Số CpG features
        embed_dim:       Chiều embedding chung (256)
        num_classes:     Số subtype (5)
        self_attn_heads: Heads cho self-attention (8)
        cross_attn_heads:Heads cho cross-attention (32)
        dropout_encoder: Dropout trong encoder (0.1)
        dropout_clf:     Dropout trong classifier (0.3)
        hidden_dim:      Chiều hidden layer classifier (128)
    """

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

        # (a) Modality-specific self-attention encoders
        self.gene_encoder   = ModalityEncoder(gene_dim,   embed_dim, self_attn_heads, dropout_encoder)
        self.mirna_encoder  = ModalityEncoder(mirna_dim,  embed_dim, self_attn_heads, dropout_encoder)
        self.methyl_encoder = ModalityEncoder(methyl_dim, embed_dim, self_attn_heads, dropout_encoder)

        # (b) Cross-attention fusion
        self.fusion = CrossAttentionFusion(embed_dim, cross_attn_heads, dropout_encoder)

        # (c) Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_clf),
            nn.Linear(hidden_dim, num_classes),
        )

        # Loss function
        self.focal_loss = FocalLoss(gamma=2.0, alpha=1.0, num_classes=num_classes)

        # Lưu config để dễ reconstruct
        self.config = dict(
            gene_dim=gene_dim, mirna_dim=mirna_dim, methyl_dim=methyl_dim,
            embed_dim=embed_dim, num_classes=num_classes,
            self_attn_heads=self_attn_heads, cross_attn_heads=cross_attn_heads,
            dropout_encoder=dropout_encoder, dropout_clf=dropout_clf,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        x_gene:   torch.Tensor,
        x_mirna:  torch.Tensor,
        x_methyl: torch.Tensor,
    ):
        """
        Args:
            x_gene:   (B, gene_dim)   — log2(TPM+1) normalized
            x_mirna:  (B, mirna_dim)  — log2(RPM+1) from GDC
            x_methyl: (B, methyl_dim) — beta values [0, 1]
        Returns:
            logits:  (B, num_classes)
            w:       (3,) modality weights [gene, mirna, methyl]
        """
        # (a) Encode từng modality
        z_gene   = self.gene_encoder(x_gene)      # (B, embed_dim)
        z_mirna  = self.mirna_encoder(x_mirna)    # (B, embed_dim)
        z_methyl = self.methyl_encoder(x_methyl)  # (B, embed_dim)

        # (b) Cross-attention fusion
        f_final, w = self.fusion(z_gene, z_mirna, z_methyl)  # (B, embed_dim)

        # (c) Classify
        logits = self.classifier(f_final)          # (B, num_classes)

        return logits, w

    def compute_loss(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
        w:       torch.Tensor,
        lambda1: float = 0.01,
        lambda2: float = 0.01,
    ):
        """
        Tính total loss: L = L_focal + λ1‖w-1/3‖² + λ2‖W_c‖²_F

        Args:
            logits:  (B, num_classes)
            targets: (B,) — nhãn 0-indexed
            w:       (3,) modality weights
            lambda1: hệ số regularization modality weights
            lambda2: hệ số Frobenius norm trên cross-attention weights
        Returns:
            total_loss:  scalar
            focal_loss:  scalar (để log riêng)
        """
        # Focal loss
        loss_focal = self.focal_loss(logits, targets)

        # λ1: push modality weights toward uniform distribution (1/3 mỗi cái)
        loss_modality = lambda1 * torch.sum((w - 1.0 / 3) ** 2)

        # λ2: Frobenius norm trên in_proj_weight của cross-attention
        # in_proj_weight shape: (3*embed_dim, embed_dim) gộp Q/K/V projections
        w_c = self.fusion.cross_attn.in_proj_weight
        loss_frob = lambda2 * torch.norm(w_c, p='fro') ** 2

        total_loss = loss_focal + loss_modality + loss_frob
        return total_loss, loss_focal

    def get_modality_weights(self) -> dict:
        """Trả về modality weights hiện tại (sau softmax, detach từ graph)."""
        w = F.softmax(self.fusion.modality_logits, dim=0).detach().cpu()
        return {
            'Gene':        round(w[0].item(), 4),
            'miRNA':       round(w[1].item(), 4),
            'Methylation': round(w[2].item(), 4),
        }

    def count_parameters(self) -> int:
        """Đếm số trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 5. QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Dimensions từ GDC Harmonized data (cập nhật nếu TPM cho số khác)
    GENE_DIM   = 19962
    MIRNA_DIM  = 1881
    METHYL_DIM = 23111
    BATCH_SIZE = 32

    print("Khởi tạo MoXGATE model...")
    model = MoXGATE(
        gene_dim   = GENE_DIM,
        mirna_dim  = MIRNA_DIM,
        methyl_dim = METHYL_DIM,
    )

    print(f"Số trainable parameters: {model.count_parameters():,}")
    print(f"Config: {model.config}\n")

    # Tạo dummy input
    x_gene   = torch.randn(BATCH_SIZE, GENE_DIM)
    x_mirna  = torch.randn(BATCH_SIZE, MIRNA_DIM)
    x_methyl = torch.randn(BATCH_SIZE, METHYL_DIM)
    targets  = torch.randint(0, 5, (BATCH_SIZE,))

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, w = model(x_gene, x_mirna, x_methyl)

    print(f"Input  — Gene: {x_gene.shape}, miRNA: {x_mirna.shape}, Methyl: {x_methyl.shape}")
    print(f"Output — logits: {logits.shape}, modality_weights: {w}")
    print(f"Modality weights: {model.get_modality_weights()}")

    # Loss check
    model.train()
    logits, w = model(x_gene, x_mirna, x_methyl)
    total_loss, focal_loss = model.compute_loss(logits, targets, w)
    print(f"\nLoss — total: {total_loss.item():.4f}, focal: {focal_loss.item():.4f}")
    print("\n✓ Sanity check passed!")
