"""
preprocess_Gene.py
==================
Xử lý dữ liệu Gene Expression (STAR counts) từ TCGA GDC cho GIAC dataset.

Nguồn dữ liệu:
    GDC TCGA Harmonized — STAR - Counts
    (COAD n=514, ESCA n=198, READ n=177, STAD n=477 — trước khi lọc -01)

Pipeline:
    ┌─────────────────────────────────────────────────────────────────┐
    │ BƯỚC 1 — Annotation                                             │
    │   Đọc GENCODE GTF → lấy gene_id của protein-coding genes        │
    │   (~19,962 genes với GENCODE v36)                               │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 2 — Xử lý từng cancer type (lặp qua 4 file TSV)            │
    │   2a. Lọc protein-coding genes (theo gene_id ENSG có version)   │
    │   2b. Lọc mẫu khối u nguyên phát: barcode[13:15] == '01'        │
    │       (loại normal tissue, recurrence, metastasis)              │
    │   2c. Rút gọn barcode → 12 ký tự Patient ID                     │
    │       TCGA-AA-3819-01A-... → TCGA-AA-3819                       │
    │   2d. Loại bỏ bệnh nhân trùng lặp (giữ Vial A = first)          │
    │   2e. Transpose → (Bệnh nhân × Genes)                           │
    │   2f. Log2(x+1) normalization                                   │
    │       STAR cho raw counts → cần normalize trước khi dùng ML     │
    │       log2(x+1): +1 để tránh log(0); chuẩn trong RNA-seq        │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 3 — Gộp 4 cancer types                                     │
    │   pd.concat(join='inner') → chỉ giữ genes có ở tất cả cohorts   │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 4 — Quality Control                                        │
    │   Loại gene thiếu dữ liệu ở >40% bệnh nhân                      │
    │   Điền phần còn lại bằng Median Imputation                      │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 5 — Lưu kết quả                                            │
    │   → processed_gene.csv  (Bệnh nhân × Genes)                     │
    └─────────────────────────────────────────────────────────────────┘

Lưu ý:
    - Dữ liệu GDC Harmonized, không cố khớp con số paper.
    - Paper dùng RSEM normalized FPKM (legacy pipeline) → số gene lệch nhẹ.
    - Kết quả thực tế phụ thuộc vào số sample có trong GDC tại thời điểm tải.
"""

import os
import glob
import gzip
import argparse
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 1. ĐỌC GTF → DANH SÁCH PROTEIN-CODING GENES
# ─────────────────────────────────────────────

def get_protein_coding_genes(gtf_path: str) -> set:
    """
    Đọc file GENCODE GTF (plain hoặc .gz), trả về set các gene_id
    của protein-coding genes (dạng có version: ENSG00000000003.15).

    Args:
        gtf_path: Đường dẫn tới file GTF hoặc GTF.gz.

    Returns:
        Set các Ensembl gene ID (có version number).
    """
    coding_genes = set()
    print(f"[Gene] Đọc GTF annotation: {os.path.basename(gtf_path)}")

    # Hỗ trợ cả file nén (.gz) và file thường
    open_func = gzip.open if gtf_path.endswith('.gz') else open
    mode = 'rt'

    with open_func(gtf_path, mode) as f:
        for line in f:
            # Bỏ dòng comment
            if line.startswith('#'):
                continue

            fields = line.split('\t')
            if len(fields) < 9:
                continue

            # Chỉ xử lý dòng 'gene' (không phải transcript/exon/CDS)
            if fields[2] != 'gene':
                continue

            attrs = fields[8]

            # Chỉ lấy protein_coding
            if 'gene_type "protein_coding"' not in attrs:
                continue

            # Trích gene_id (có version, vd: ENSG00000000003.15)
            for part in attrs.split(';'):
                part = part.strip()
                if part.startswith('gene_id'):
                    gene_id = part.split('"')[1]
                    coding_genes.add(gene_id)
                    break

    print(f"[Gene] ✓ Tìm thấy {len(coding_genes):,} protein-coding genes trong GTF")
    return coding_genes


# ─────────────────────────────────────────────
# 2. ĐỌC VÀ XỬ LÝ TỪNG FILE TSV
# ─────────────────────────────────────────────

def clean_and_transpose(file_path: str, coding_genes: set) -> pd.DataFrame:
    """
    Đọc 1 file TSV gene expression (STAR raw counts, dạng matrix):
        - Rows: Ensembl gene ID có version (ENSG00000000003.15)
        - Cols: TCGA barcodes đầy đủ (TCGA-DC-6683-01A-11R-...)
        - Values: raw read counts (integer)

    Pipeline:
        2a. Lọc protein-coding genes theo GTF annotation
        2b. Lọc mẫu khối u nguyên phát (barcode[13:15] == '01')
        2c. Rút gọn barcode → 12 ký tự Patient ID
        2d. Loại bỏ bệnh nhân trùng lặp (giữ Vial A = first)
        2e. Transpose → (Bệnh nhân × Genes)
        2f. Log2(x+1) normalization — chuyển raw counts sang log-scale

    Args:
        file_path:    Đường dẫn file TSV.
        coding_genes: Set Ensembl gene ID protein-coding (có version).

    Returns:
        DataFrame shape (n_patients, n_coding_genes), đã log2(x+1) normalize.
    """
    cancer_name = os.path.basename(file_path)
    print(f"  -> Đọc: {cancer_name}")

    df = pd.read_csv(file_path, sep='\t', index_col=0, engine='c')

    # --- Bước 2a: Lọc protein-coding genes ---
    # Index dạng ENSG00000000003.15 — khớp trực tiếp với coding_genes từ GTF
    n_before = len(df)
    df = df[df.index.isin(coding_genes)]
    print(f"     Lọc protein-coding: {n_before:,} → {len(df):,} genes")

    # --- Bước 2b: Lọc mẫu khối u nguyên phát (-01) ---
    # TCGA barcode: TCGA-XX-XXXX-01A-... → ký tự [13:15] là sample type
    # '01' = Primary Solid Tumor, '11' = Normal, '06' = Metastatic, v.v.
    tumor_cols = [
        col for col in df.columns
        if isinstance(col, str) and len(col) >= 15 and col[13:15] == '01'
    ]
    df_tumor = df[tumor_cols]

    # --- Bước 2c: Rút gọn barcode → 12 ký tự Patient ID ---
    # TCGA-DC-6683-01A-11R-... → TCGA-DC-6683
    df_tumor.columns = [col[:12] for col in df_tumor.columns]

    # --- Bước 2d: Loại bỏ bệnh nhân trùng lặp ---
    # Cùng 1 bệnh nhân có thể có nhiều aliquot → giữ Vial A (xuất hiện đầu tiên)
    df_tumor = df_tumor.loc[:, ~df_tumor.columns.duplicated(keep='first')]
    print(f"     Bệnh nhân sau lọc: {df_tumor.shape[1]}")

    # --- Bước 2e: Transpose → (Bệnh nhân × Genes) ---
    df_t = df_tumor.T

    # --- Bước 2f: Log2(x+1) normalization ---
    # STAR counts là raw integer counts → phân phối rất lệch (skewed)
    # log2(x+1): +1 tránh log(0) với gene không biểu hiện
    # Kết quả: phân phối gần normal hơn, phù hợp cho downstream ML/DL
    df_t = np.log2(df_t + 1)

    return df_t


# ─────────────────────────────────────────────
# 3. XỬ LÝ MISSING VALUES
# ─────────────────────────────────────────────

def handle_missing_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 4 — Quality Control trên ma trận tổng:
        4a. Loại gene thiếu dữ liệu ở >40% bệnh nhân
            (gene không đo được ở quá nhiều bệnh nhân → không đáng tin cậy)
        4b. Median Imputation cho phần còn lại
            (điền giá trị trung vị của gene đó trên toàn cohort)

    Args:
        df: DataFrame (n_patients, n_genes).

    Returns:
        DataFrame đã lọc và impute.
    """
    print(f"\n[Gene] Kích thước trước QC: {df.shape} (Bệnh nhân × Genes)")

    # 4a: Gene hợp lệ nếu có giá trị ở ít nhất 60% bệnh nhân
    min_valid_count = int(len(df) * 0.60)
    df_filtered = df.dropna(axis=1, thresh=min_valid_count)
    n_dropped = df.shape[1] - df_filtered.shape[1]
    if n_dropped > 0:
        print(f"[Gene]  Loại {n_dropped:,} genes thiếu >40% → còn {df_filtered.shape[1]:,} genes")

    # 4b: Điền khuyết bằng Median của từng gene
    df_imputed = df_filtered.fillna(df_filtered.median())

    print(f"[Gene] Kích thước sau QC: {df_imputed.shape}")
    return df_imputed


# ─────────────────────────────────────────────
# 4. HÀM CHÍNH: PROCESS_GENE
# ─────────────────────────────────────────────

def process_gene(input_dir: str, output_dir: str, gtf_path: str) -> pd.DataFrame:
    """
    Toàn bộ pipeline xử lý Gene Expression (STAR counts → log2 normalized).

    Args:
        input_dir:  Thư mục gốc chứa dữ liệu omics (subfolder 'gene/' bên trong)
        output_dir: Thư mục lưu output.
        gtf_path:   Đường dẫn file GENCODE GTF (plain hoặc .gz).

    Returns:
        DataFrame hoàn chỉnh (Bệnh nhân × Genes), đồng thời lưu ra CSV.
    """
    print("\n" + "="*60)
    print("  BẮT ĐẦU XỬ LÝ GENE EXPRESSION (STAR Counts → Log2)")
    print("  Nguồn: GDC TCGA Harmonized")
    print("="*60)

    # ── Bước 1: Lấy danh sách protein-coding genes từ GTF ──────────────
    coding_genes = get_protein_coding_genes(gtf_path)

    # ── Bước 2: Đọc và xử lý từng file TSV ────────────────────────────
    gene_dir = os.path.join(input_dir, "gene")
    file_paths = glob.glob(os.path.join(gene_dir, "*.tsv"))

    if not file_paths:
        raise FileNotFoundError(
            f"Không tìm thấy file TSV nào trong: {gene_dir}\n"
            f"Kiểm tra lại cấu trúc thư mục: input_dir/gene/*.tsv"
        )

    print(f"\n[Gene] Tìm thấy {len(file_paths)} file TSV:")
    df_list = []
    for fp in sorted(file_paths):
        df_cancer = clean_and_transpose(fp, coding_genes)
        df_list.append(df_cancer)

    # ── Bước 3: Gộp 4 cancer types ─────────────────────────────────────
    # join='inner': chỉ giữ genes xuất hiện ở tất cả 4 cohorts
    # (đảm bảo ma trận không có NaN do khác nhau về gene coverage giữa cohorts)
    print(f"\n[Gene] Gộp {len(df_list)} cancer types (join='inner')...")
    master_df = pd.concat(df_list, axis=0, join='inner')
    print(f"[Gene] Sau khi gộp: {master_df.shape} (Bệnh nhân × Genes)")

    # ── Bước 4: Quality Control ─────────────────────────────────────────
    final_df = handle_missing_and_impute(master_df)

    # ── Bước 5: Lưu output ──────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "processed_gene.csv")
    final_df.to_csv(out_path)
    print(f"\n[Gene] ✓ Đã lưu: {out_path}")
    print(f"[Gene] ✓ Shape cuối: {final_df.shape}  (Bệnh nhân × Genes, log2(x+1) normalized)")
    print("="*60)

    return final_df


# ─────────────────────────────────────────────
# 5. ENTRY POINT (chạy độc lập để test)
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Xử lý Gene Expression TCGA cho GIAC dataset"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Thư mục gốc chứa dữ liệu omics (chứa subfolder 'gene/')"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Thư mục lưu file processed_gene.csv"
    )
    parser.add_argument(
        "--gtf_path", type=str, required=True,
        help="Đường dẫn file GTF annotation (plain hoặc .gz)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_gene(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        gtf_path=args.gtf_path,
    )