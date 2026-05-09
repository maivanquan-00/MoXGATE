"""
preprocess_Gene.py
==================
Xử lý dữ liệu Gene Expression từ TCGA RNAseq - STAR - TPM cho GIAC dataset.

Nguồn dữ liệu:
    TCGA RNAseq - STAR - TPM
    Dataset đầu vào đã ở dạng log2(TPM + 1)
    → KHÔNG cần áp dụng log transform thêm lần nữa
    (nếu log thêm sẽ bị double-log và nén range quá mạnh)

Pipeline:
    ┌─────────────────────────────────────────────────────────────────┐
    │ BƯỚC 1 — Annotation                                             │
    │   Đọc GENCODE GTF → lấy gene_id của protein-coding genes        │
    │   (~19,962 genes với GENCODE v36)                               │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 2 — Xử lý từng cancer type (lặp qua các file TSV)          │
    │   2a. Lọc protein-coding genes (theo gene_id ENSG có version)   │
    │   2b. Lọc mẫu khối u nguyên phát: barcode[13:15] == '01'        │
    │       (loại normal tissue, recurrence, metastasis)              │
    │   2c. Rút gọn barcode → 12 ký tự Patient ID                     │
    │       TCGA-AA-3819-01A-... → TCGA-AA-3819                       │
    │   2d. Loại bỏ bệnh nhân trùng lặp (giữ Vial A = first)          │
    │   2e. Transpose → (Bệnh nhân × Genes)                           │
    │       (KHÔNG log transform — Xena đã log2(norm_count+1))        │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 3 — Gộp các cancer types                                   │
    │   pd.concat(join='inner') → chỉ giữ genes có ở tất cả cohorts   │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 4 — Quality Control                                        │
    │   Loại gene thiếu dữ liệu ở >40% bệnh nhân                      │
    │   Điền phần còn lại bằng Median Imputation                      │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 5 — Map Ensembl → gene symbol (HGNC complete set)          │
    │   ENSG00000000003.15 → TSPAN6 (drop unmapped, dedup symbol)     │
    │   Symbol bắt buộc cho graph PPI/Reactome edges                  │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 6 — Lưu kết quả                                            │
    │   → processed_gene.csv  (Bệnh nhân × gene symbols)              │
    └─────────────────────────────────────────────────────────────────┘

Sanity check sau preprocessing:
    Giá trị log2(norm_count+1) thường có max ~ 15-20, median ~ 5-8.
    Nếu max ~ 4 và median ~ 2 → ĐÃ BỊ DOUBLE LOG (bug cũ).
"""

import os
import glob
import gzip
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
    Đọc 1 file TSV gene expression (Xena HiSeqV2, dạng matrix):
        - Rows: Ensembl gene ID có version (ENSG00000000003.15)
        - Cols: TCGA barcodes đầy đủ (TCGA-DC-6683-01A-11R-...)
        - Values: log2(norm_count + 1) — Xena đã chuẩn hoá sẵn

    Pipeline:
        2a. Lọc protein-coding genes theo GTF annotation
        2b. Lọc mẫu khối u nguyên phát (barcode[13:15] == '01')
        2c. Rút gọn barcode → 12 ký tự Patient ID
        2d. Loại bỏ bệnh nhân trùng lặp (giữ Vial A = first)
        2e. Transpose → (Bệnh nhân × Genes)

    Args:
        file_path:    Đường dẫn file TSV.
        coding_genes: Set Ensembl gene ID protein-coding (có version).

    Returns:
        DataFrame shape (n_patients, n_coding_genes); values giữ nguyên
        log2(TPM+1) từ nguồn RNAseq STAR (KHÔNG log thêm).
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
    # KHÔNG log transform — dữ liệu đã ở dạng log2(TPM+1).
    # Bug cũ: áp np.log2(x+1) thêm lần nữa → double-log nén range quá mạnh.
    return df_tumor.T


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

def _load_hgnc_mapping(hgnc_path: str) -> dict:
    """Đọc hgnc_complete_set.txt → dict {ensembl_gene_id (no version): symbol}."""
    hgnc = pd.read_csv(
        hgnc_path, sep="\t",
        usecols=["ensembl_gene_id", "symbol"],
        low_memory=False, dtype=str,
    )
    hgnc = hgnc.dropna(subset=["ensembl_gene_id", "symbol"])
    hgnc["ensembl_gene_id"] = hgnc["ensembl_gene_id"].str.strip()
    hgnc["symbol"]          = hgnc["symbol"].str.strip()
    hgnc = hgnc.drop_duplicates(subset="ensembl_gene_id", keep="first")
    return dict(zip(hgnc["ensembl_gene_id"], hgnc["symbol"]))


def _map_ensembl_to_symbol(df: pd.DataFrame, ensg_to_sym: dict) -> pd.DataFrame:
    """Chuyển cột Ensembl ID (có version) → gene symbol. Drop unmapped, dedup."""
    new_cols = [ensg_to_sym.get(c.split(".")[0]) for c in df.columns]
    mask = [s is not None for s in new_cols]
    n_mapped = sum(mask)
    print(f"[Gene] Map Ensembl→symbol: {n_mapped:,}/{len(new_cols):,} genes")

    df = df.loc[:, mask].copy()
    df.columns = [s for s in new_cols if s is not None]

    n_before = df.shape[1]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    if df.shape[1] < n_before:
        print(f"[Gene] Loại {n_before - df.shape[1]:,} cột symbol trùng (giữ cái đầu)")

    return df


def process_gene(
    input_dir: str,
    output_dir: str,
    gtf_path: str,
    hgnc_path: str | None = None,
) -> pd.DataFrame:
    """
    Toàn bộ pipeline xử lý Gene Expression: log2(TPM+1) → symbol matrix.

    Args:
        input_dir:  Thư mục gốc chứa dữ liệu omics (subfolder 'gene/' bên trong)
        output_dir: Thư mục lưu output.
        gtf_path:   Đường dẫn file GENCODE GTF (plain hoặc .gz).
        hgnc_path:  (Tùy chọn) hgnc_complete_set.txt — nếu có, map Ensembl → symbol.
                    Nếu None hoặc file không tồn tại: giữ Ensembl ID (graph PPI/Reactome
                    sẽ rỗng — chỉ dùng khi debug).

    Returns:
        DataFrame (Bệnh nhân × Genes), cột là gene symbol nếu có hgnc_path.
    """
    print("\n" + "="*60)
    print("  BẮT ĐẦU XỬ LÝ GENE EXPRESSION (RNAseq - STAR - TPM)")
    print("  Nguồn: dữ liệu đã ở dạng log2(TPM+1)")
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

    # ── Bước 3: Gộp các cancer types ───────────────────────────────────
    # join='inner': chỉ giữ genes xuất hiện ở tất cả cohorts
    # (đảm bảo ma trận không có NaN do khác nhau về gene coverage giữa cohorts)
    print(f"\n[Gene] Gộp {len(df_list)} cancer types (join='inner')...")
    master_df = pd.concat(df_list, axis=0, join='inner')
    print(f"[Gene] Sau khi gộp: {master_df.shape} (Bệnh nhân × Genes)")

    # ── Bước 4: Quality Control ─────────────────────────────────────────
    final_df = handle_missing_and_impute(master_df)

    # ── Bước 5: Map Ensembl → gene symbol (nếu có HGNC) ─────────────────
    if hgnc_path and os.path.exists(hgnc_path):
        ensg_to_sym = _load_hgnc_mapping(hgnc_path)
        print(f"[Gene] HGNC: {len(ensg_to_sym):,} mappings")
        final_df = _map_ensembl_to_symbol(final_df, ensg_to_sym)
        print(f"[Gene] Sau khi map symbol: {final_df.shape}")
    else:
        print(f"[Gene] ⚠️  Không có HGNC path → giữ Ensembl ID (graph PPI/Reactome sẽ rỗng!)")

    # ── Bước 6: Lưu output ──────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "processed_gene.csv")
    final_df.to_csv(out_path)
    print(f"\n[Gene] ✓ Đã lưu: {out_path}")
    print(f"[Gene] ✓ Shape cuối: {final_df.shape}  (Bệnh nhân × Genes)")

    # Sanity check: phát hiện double-log bug (giá trị max thấp bất thường)
    max_val    = float(final_df.values.max())
    median_val = float(pd.Series(final_df.values.flatten()).median())
    print(f"[Gene] ✓ Sanity check: max={max_val:.2f}  median={median_val:.2f}")
    if max_val < 6.0:
        print(f"[Gene] ⚠️  CẢNH BÁO: max={max_val:.2f} thấp bất thường — có thể đã bị double-log!")
        print(f"[Gene]    Kỳ vọng: max ~ 15-20 với log2(norm_count+1) chuẩn của Xena.")
    print("="*60)

    return final_df

