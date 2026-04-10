
---

## KẾ HOẠCH THỰC HIỆN CHI TIẾT — 12 TUẦN

### Tổng quan 4 Module cải tiến

| # | Module | Mô tả | Effort | Rủi ro |
|---|--------|-------|--------|--------|
| 3 | Sparse Cross-Attention | Thay softmax → sparsemax trong cross-attention fusion | 1 tuần | Thấp |
| 1 | Heterogeneous Graph | Xây đồ thị sinh học từ STRING PPI + miRTarBase + Manifest | 3 tuần | Trung bình |
| 2 | GAT Encoder | Thay self-attention bằng Graph Attention Network | 5 tuần | Cao |
| 4 | SHAP + GSEA | Giải thích sinh học per-patient + pathway enrichment | 3 tuần | Thấp |

> **Thứ tự thực hiện**: Module 3 → 1 → 2 → 4 (ưu tiên rủi ro thấp trước)

---

### Tuần 1 (10-17/4) — Module 3: Sparse Cross-Attention

| Ngày | Task | Deliverable |
|------|------|-------------|
| 1-2 | Đọc paper Sparsemax (Martins & Astudillo, ICML 2016) + α-entmax (Peters et al., ACL 2019) | Hiểu thuật toán projection onto simplex |
| 3-4 | `pip install entmax`, thay `F.softmax` → `sparsemax` trong `CrossAttentionFusion` | Code chạy, train thành công |
| 5-6 | Chạy ablation: softmax vs sparsemax vs α-entmax(1.5) | Bảng so sánh accuracy/F1 |
| 7 | Viết kết quả, commit | ✅ **Milestone: Sparsemax done** |

---

### Tuần 2-4 (17/4 - 8/5) — Module 1: Heterogeneous Graph

| Tuần | Task | Chi tiết |
|------|------|---------|
| T2 | Download + parse dữ liệu | STRING: `9606.protein.links.v12.0.txt.gz` (filter score≥700). miRTarBase: `hsa_MTI.xlsx` (validated interactions) |
| T2-3 | Mapping gene ID | TCGA dùng ENSEMBL (ENSG), STRING dùng ENSP, miRTarBase dùng gene symbol. Dùng `pyensembl` hoặc `mygene` để convert |
| T3 | CpG → Gene mapping | Parse `HumanMethylation450_manifest.csv` (có sẵn trong `data_original/annotation/`). Cột `UCSC_RefGene_Name` cho CpG→gene link |
| T3-4 | Build PyG HeteroData | 3 node types: `gene`, `cpg`, `mirna`. Edge types: `(gene, ppi, gene)`, `(mirna, targets, gene)`, `(cpg, regulates, gene)` |
| T4 | Validate + thống kê | In số nodes, edges per type, degree distribution. Sanity check: gene count khớp với data |

**Output**: `build_graph.py` + `data_graph/hetero_graph.pt`

**Backup**: Nếu methyl-QTL quá khó → dùng CpG→gene từ manifest annotation thay thế.

---

### Tuần 5-8 (8/5 - 10/6) — Module 2: GAT Encoder

| Tuần | Task | Chi tiết |
|------|------|---------|
| T5 | Đọc GAT (Veličković, ICLR 2018) + GATv2 (Brody, ICLR 2022) | Hiểu attention coefficient α_ij |
| T5-6 | **Pilot**: GAT chỉ trên gene modality | Thay `ModalityEncoder` gene bằng 2-layer GAT (PyG `GATConv`). So sánh accuracy |
| T6-7 | HeteroGAT đầy đủ 3 modalities | Dùng `HeteroConv` wrapper: mỗi edge type có weight riêng |
| T7 | Tích hợp với Module 3 | GAT output → Sparse Cross-Attention → Classifier. End-to-end training |
| T8 | Tuning + ablation study | Số GAT layers (2-3), heads (4-8), hidden dim |

> **⚠️ CHECKPOINT QUYẾT ĐỊNH (cuối tuần 6):** Nếu pilot GAT trên gene đơn lẻ không chạy hoặc kết quả kém hơn baseline → **DỪNG Module 2**, chuyển sang Module 4. Report GAT như "hướng phát triển" trong luận văn.

---

### Tuần 9-11 (11/6 - 1/7) — Module 4: SHAP + GSEA

| Tuần | Task | Chi tiết |
|------|------|---------|
| T9 | Setup Captum | `pip install captum`. Wrap model cho `IntegratedGradients`. Chạy attribution trên test set (79 patients) |
| T9-10 | Feature importance | Per-patient: top-50 gene, top-50 CpG, top-20 miRNA. Per-subtype: aggregate top features |
| T10-11 | GSEA | `pip install gseapy`. Input top gene list → enrichment against KEGG/Reactome. Kỳ vọng: EBV+ → DNA methylation pathways, MSI → mismatch repair |
| T11 | Case study | 2-3 bệnh nhân cụ thể: "Patient X phân loại MSI, top gene: MLH1, MSH2, MSH6 — đúng với literature" |

**Output**: Bảng top features per subtype + enrichment results + case study figures

---

### Tuần 6-12 (song song) — Viết luận văn

| Tuần | Chương | Nội dung |
|------|--------|---------|
| T6-7 | Phương pháp | Mô tả 4 module, công thức toán, kiến trúc diagram |
| T8-9 | Kết quả | Bảng so sánh, ablation study, confusion matrix |
| T11 | Giải thích sinh học | GSEA results, case study, so sánh Liu et al. 2018 |
| T12 | Tổng hợp + review | Mở đầu, kết luận, chỉnh sửa, format |

> 🎯 **Deadline nộp luận văn: ~10/7/2026**

---

### Cấu trúc thư mục dự kiến

```
MoXGATE/
├── data_original/          ← dữ liệu gốc TCGA
├── data_final/             ← dữ liệu đã xử lý
├── data_processed/         ← dữ liệu trung gian
├── data_graph/             ← MỚI: graph data
│   ├── string_ppi.txt
│   ├── mirtarbase.txt
│   └── hetero_graph.pt
├── model.py                ← baseline MoXGATE
├── model_sparse.py         ← Module 3: + Sparsemax
├── model_gat.py            ← Module 2: + GAT encoder
├── build_graph.py          ← Module 1: graph construction
├── interpret.py            ← Module 4: SHAP + GSEA
├── train.py                ← training script
├── dataset.py              ← data loading + split
├── ablation.py             ← ablation study runner
└── README.md
```

---

### Nguyên tắc an toàn

1. **Mỗi module commit riêng** — luôn giữ baseline chạy được
2. **Checkpoint tuần 6** — quyết định tiếp Module 2 hay dừng
3. **Viết luận văn song song** từ tuần 6 — không để dồn cuối
4. **Mỗi module có ablation riêng** — kể cả kết quả xấu cũng report được

---

### Baseline hiện tại

```
MoXGATE (original paper): 95% accuracy (paper claimed)
MoXGATE (reimplemented):  92.4% accuracy
  - CIN:    96% recall (74 samples)
  - MSI:    100% recall, 50% precision (2 samples)
  - GS:     0% (1 sample — quá ít)
  - HM-SNV: 0% (2 samples — quá ít)

Fixes đã áp dụng:
  - λ2: 0.01 → 1e-4 (Frobenius norm không thống trị loss)
  - Stratified val split (đảm bảo subtypes đại diện trong val)
  - α=1 đúng paper (Section 2.1.4)
```

