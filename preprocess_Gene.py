"""
preprocess_Gene.py
==================
Xử lý dữ liệu Gene Expression (STAR counts) từ TCGA cho GIAC dataset.

Pipeline:
    1. Đọc file GTF → lấy danh sách protein-coding genes (~20,530 genes)
    2. Với mỗi file TSV (1 cancer type):
        - Lọc protein-coding genes
        - Lọc mẫu khối u nguyên phát (barcode suffix -01)
        - Rút gọn barcode về 12 ký tự (Patient ID)
        - Loại bỏ bệnh nhân trùng lặp (giữ Vial A)
        - Transpose → (Bệnh nhân x Genes)
    3. Gộp 4 cancer types → lấy giao thoa features (join='inner')
    4. Lọc features thiếu > 40% → Median Imputation
    5. Lưu ra processed_gene.csv

Kết quả kỳ vọng: ~(1220, 20530) — khớp với paper MoXGATE
"""

import os
import glob
import gzip
import argparse
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
    Đọc 1 file TSV gene expression (STAR counts dạng matrix):
        - Rows: Ensembl gene ID (có version)
        - Cols: TCGA barcodes (vd: TCGA-DC-6683-01A-...)

    Thực hiện:
        1. Lọc protein-coding genes
        2. Lọc mẫu khối u nguyên phát (-01)
        3. Rút gọn barcode → 12 ký tự Patient ID
        4. Loại bỏ trùng lặp (giữ Vial A - first)
        5. Transpose → (Bệnh nhân x Genes)

    Args:
        file_path:     Đường dẫn file TSV.
        coding_genes:  Set Ensembl ID protein-coding (có version).

    Returns:
        DataFrame shape (n_patients, n_coding_genes).
    """
    cancer_name = os.path.basename(file_path)
    print(f"  -> Đọc: {cancer_name}")

    df = pd.read_csv(file_path, sep='\t', index_col=0, engine='c')

    # --- Bước 1: Lọc protein-coding genes ---
    # Index dạng ENSG00000000003.15 — khớp trực tiếp với coding_genes
    n_before = len(df)
    df = df[df.index.isin(coding_genes)]
    print(f"     Lọc protein-coding: {n_before:,} → {len(df):,} genes")

    # --- Bước 2: Lọc mẫu khối u nguyên phát (-01) ---
    # TCGA barcode: TCGA-XX-XXXX-01A-... → vị trí [13:15] == '01'
    tumor_cols = [
        col for col in df.columns
        if isinstance(col, str) and len(col) >= 15 and col[13:15] == '01'
    ]
    df_tumor = df[tumor_cols]

    # --- Bước 3: Rút gọn barcode → 12 ký tự Patient ID ---
    # TCGA-DC-6683-01A → TCGA-DC-6683
    df_tumor.columns = [col[:12] for col in df_tumor.columns]

    # --- Bước 4: Loại bỏ bệnh nhân trùng lặp (giữ Vial A - first) ---
    df_tumor = df_tumor.loc[:, ~df_tumor.columns.duplicated(keep='first')]

    print(f"     Bệnh nhân sau lọc: {df_tumor.shape[1]}")

    # --- Bước 5: Transpose → (Bệnh nhân x Genes) ---
    return df_tumor.T


# ─────────────────────────────────────────────
# 3. XỬ LÝ MISSING VALUES
# ─────────────────────────────────────────────

def handle_missing_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý missing values trên ma trận tổng:
        1. Loại features (cột) thiếu > 40% dữ liệu (thresh = 60% bệnh nhân)
        2. Điền phần còn lại bằng Median Imputation

    Args:
        df: DataFrame (n_patients, n_features).

    Returns:
        DataFrame đã lọc và impute.
    """
    print(f"\n[Gene] Kích thước trước lọc: {df.shape} (Bệnh nhân x Features)")

    # Ngưỡng: feature hợp lệ nếu có ít nhất 60% bệnh nhân có giá trị
    min_valid_count = int(len(df) * 0.60)
    df_filtered = df.dropna(axis=1, thresh=min_valid_count)

    # Điền khuyết bằng Median
    df_imputed = df_filtered.fillna(df_filtered.median())

    print(f"[Gene] Kích thước sau lọc & impute: {df_imputed.shape}")
    return df_imputed


# ─────────────────────────────────────────────
# 4. HÀM CHÍNH: PROCESS_GENE
# ─────────────────────────────────────────────

def process_gene(input_dir: str, output_dir: str, gtf_path: str) -> pd.DataFrame:
    """
    Toàn bộ pipeline xử lý Gene Expression.

    Args:
        input_dir:  Thư mục chứa các file TSV gene (vd: .../multi_omics/gene/)
        output_dir: Thư mục lưu output.
        gtf_path:   Đường dẫn file GTF (plain hoặc .gz).

    Returns:
        DataFrame đã xử lý hoàn chỉnh, đồng thời lưu ra CSV.
    """
    print("\n" + "="*60)
    print("  BẮT ĐẦU XỬ LÝ GENE EXPRESSION")
    print("="*60)

    # Bước 1: Lấy danh sách protein-coding genes từ GTF
    coding_genes = get_protein_coding_genes(gtf_path)

    # Bước 2: Đọc và xử lý từng file TSV
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

    # Bước 3: Gộp 4 cancer types, lấy giao thoa features
    print(f"\n[Gene] Gộp {len(df_list)} cancer types (join='inner' — lấy genes chung)...")
    master_df = pd.concat(df_list, axis=0, join='inner')
    print(f"[Gene] Sau khi gộp: {master_df.shape} (Bệnh nhân x Genes)")

    # Bước 4: Xử lý missing values
    final_df = handle_missing_and_impute(master_df)

    # Bước 5: Lưu output
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "processed_gene.csv")
    final_df.to_csv(out_path)
    print(f"\n[Gene] ✓ Đã lưu: {out_path}")
    print(f"[Gene] ✓ Shape cuối: {final_df.shape}  (kỳ vọng: ~1220 x 20530)")
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