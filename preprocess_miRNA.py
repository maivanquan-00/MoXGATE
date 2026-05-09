"""
preprocess_miRNA.py
===================
Xử lý dữ liệu miRNA Expression từ TCGA/GDC cho các dataset của dự án.

Nguồn dữ liệu:
    GDC / TCGA miRNA TSV (ví dụ LGG: TCGA-LGG.mirna.tsv)
    Dataset ID: stem loop expression
    Platform:   Illumina miRNA-Seq
    Naming:     stem-loop level (hsa-mir-21-1, hsa-mir-100, hsa-let-7a-1...)
    Unit:       log2(RPM + 1)
    miRBase:    v22 → ~1,881 hsa stem-loop precursors

Pipeline:
    ┌─────────────────────────────────────────────────────────────────┐
    │ BƯỚC 1 — Xử lý từng cancer type (lặp qua các file TSV)          │
    │   1a. Lọc mẫu khối u nguyên phát: barcode[13:15] == '01'        │
    │       (loại normal tissue, recurrence, metastasis)              │
    │   1b. Rút gọn barcode → 12 ký tự Patient ID                     │
    │       TCGA-AA-3819-01A-... → TCGA-AA-3819                       │
    │   1c. Loại bỏ bệnh nhân trùng lặp (giữ Vial A = first)          │
    │   1d. Transpose → (Bệnh nhân × miRNA)                           │
    │       (KHÔNG log transform — dữ liệu đã ở dạng log2(RPM+1))     │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 2 — Gộp các cancer types                                   │
    │   pd.concat(join='inner') → chỉ giữ miRNA có ở tất cả cohorts   │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 3 — Quality Control                                        │
    │   Loại miRNA thiếu dữ liệu ở >40% bệnh nhân                     │
    │   Điền phần còn lại bằng Median Imputation                      │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 4 — Lọc miRNA biểu hiện thấp                               │
    │   Giữ miRNA có log2(RPM+1) > 0.5 ở ít nhất 20% bệnh nhân         │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 5 — Lưu kết quả                                            │
    │   → processed_mirna.csv  (Bệnh nhân × miRNA)                    │
    └─────────────────────────────────────────────────────────────────┘

Sanity check sau preprocessing:
    Giá trị log2(RPM+1) thường có max ~ 15-20, nhiều giá trị 0 do miRNA
    không expressed trong tissue. Median ~ 0 là bình thường.
"""

import os
import glob
import pandas as pd

MIRNA_EXPRESSION_THRESHOLD = 0.5
MIRNA_MIN_EXPRESSION_FRACTION = 0.20


# ─────────────────────────────────────────────
# 1. ĐỌC VÀ XỬ LÝ TỪNG FILE TSV
# ─────────────────────────────────────────────

def clean_and_transpose(file_path: str) -> pd.DataFrame:
    """
    Đọc 1 file TSV miRNA expression (GDC/TCGA, dạng matrix):
        - Rows: miRNA stem-loop ID (hsa-mir-21-1, hsa-mir-100, hsa-let-7a-1, ...)
        - Cols: TCGA barcodes đầy đủ (TCGA-AA-3819-01A-...)
        - Values: log2(RPM + 1) — dữ liệu đã chuẩn hoá sẵn

    Pipeline:
        1a. Lọc mẫu khối u nguyên phát (barcode[13:15] == '01')
        1b. Rút gọn barcode → 12 ký tự Patient ID
        1c. Loại bỏ bệnh nhân trùng lặp (giữ Vial A = first)
        1d. Transpose → (Bệnh nhân × miRNA)

    Args:
        file_path: Đường dẫn file TSV.

    Returns:
        DataFrame shape (n_patients, n_mirna).
    """
    print(f"  -> Đọc: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path, sep='\t', index_col=0, engine='c')
    print(f"     Shape gốc: {df.shape[0]:,} features × {df.shape[1]:,} samples")

    # --- Bước 1a: Lọc mẫu khối u nguyên phát (-01) ---
    # TCGA barcode: TCGA-XX-XXXX-01A-... → ký tự [13:15] là sample type
    # '01' = Primary Solid Tumor, '11' = Normal, '06' = Metastatic
    tumor_cols = [
        col for col in df.columns
        if isinstance(col, str) and len(col) >= 15 and col[13:15] == '01'
    ]
    df_tumor = df[tumor_cols]

    if df_tumor.empty:
        print(f"     ⚠ Không tìm thấy cột tumor (-01)! 3 cột đầu: {list(df.columns[:3])}")
        return pd.DataFrame()

    # --- Bước 1b: Rút gọn barcode → 12 ký tự Patient ID ---
    # TCGA-AA-3819-01A-... → TCGA-AA-3819
    df_tumor.columns = [col[:12] for col in df_tumor.columns]

    # --- Bước 1c: Loại bỏ bệnh nhân trùng lặp ---
    # Cùng 1 bệnh nhân có thể có nhiều aliquot → giữ Vial A (xuất hiện đầu tiên)
    df_tumor = df_tumor.loc[:, ~df_tumor.columns.duplicated(keep='first')]
    print(f"     Bệnh nhân sau lọc: {df_tumor.shape[1]}")

    # --- Bước 1d: Transpose → (Bệnh nhân × miRNA) ---
    return df_tumor.T


# ─────────────────────────────────────────────
# 2. XỬ LÝ MISSING VALUES
# ─────────────────────────────────────────────

def handle_missing_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 3 — Quality Control trên ma trận tổng:
        3a. Loại miRNA thiếu dữ liệu ở >40% bệnh nhân
            (miRNA không đo được ở quá nhiều bệnh nhân → không đáng tin cậy)
        3b. Median Imputation cho phần còn lại
            (điền giá trị trung vị của miRNA đó trên toàn cohort)

    Args:
        df: DataFrame (n_patients, n_mirna).

    Returns:
        DataFrame đã lọc và impute.
    """
    print(f"\n[miRNA] Kích thước trước QC: {df.shape} (Bệnh nhân × miRNA)")

    # 3a: miRNA hợp lệ nếu có giá trị ở ít nhất 60% bệnh nhân
    min_valid_count = int(len(df) * 0.60)
    df_filtered = df.dropna(axis=1, thresh=min_valid_count)
    n_dropped = df.shape[1] - df_filtered.shape[1]
    if n_dropped > 0:
        print(f"[miRNA]  Loại {n_dropped:,} miRNA thiếu >40% → còn {df_filtered.shape[1]:,} miRNA")

    # 3b: Điền khuyết bằng Median của từng miRNA
    df_imputed = df_filtered.fillna(df_filtered.median())

    print(f"[miRNA] Kích thước sau QC: {df_imputed.shape}")
    return df_imputed


def filter_low_expression_mirna(
    df: pd.DataFrame,
    expression_threshold: float = MIRNA_EXPRESSION_THRESHOLD,
    min_expression_fraction: float = MIRNA_MIN_EXPRESSION_FRACTION,
) -> pd.DataFrame:
    """
    Lọc miRNA biểu hiện thấp theo rule không dùng nhãn:
        giữ miRNA nếu log2(RPM + 1) > expression_threshold
        ở ít nhất min_expression_fraction bệnh nhân.

    Rule mặc định đã test trên GIAC:
        threshold = 0.5, fraction = 20% -> khoảng 593/1881 miRNA.
    """
    expressed_fraction = (df > expression_threshold).mean(axis=0)
    keep_mask = expressed_fraction >= min_expression_fraction
    df_filtered = df.loc[:, keep_mask]

    n_dropped = df.shape[1] - df_filtered.shape[1]
    print(
        "[miRNA] Low-expression filter: "
        f"log2(RPM+1) > {expression_threshold} "
        f"ở >= {min_expression_fraction:.0%} bệnh nhân"
    )
    print(
        f"[miRNA]  Loại {n_dropped:,} miRNA biểu hiện thấp "
        f"→ còn {df_filtered.shape[1]:,} miRNA"
    )

    return df_filtered


# ─────────────────────────────────────────────
# 3. HÀM CHÍNH: PROCESS_MIRNA
# ─────────────────────────────────────────────

def process_mirna(input_dir: str, output_dir: str) -> pd.DataFrame:
    """
    Toàn bộ pipeline xử lý miRNA Expression (GDC Harmonized), có lọc
    low-expression miRNA bằng rule không dùng nhãn.

    Args:
        input_dir:  Thư mục gốc omics (chứa subfolder 'mirna/' bên trong)
        output_dir: Thư mục lưu output.

    Returns:
        DataFrame hoàn chỉnh (Bệnh nhân × miRNA), đồng thời lưu ra CSV.
    """
    print("\n" + "="*60)
    print("  BẮT ĐẦU XỬ LÝ miRNA EXPRESSION (Xena miRNA_HiSeq_gene)")
    print("  Nguồn: UCSC Xena — đã ở dạng log2(RPM+1)")
    print("="*60)

    # ── Bước 1: Đọc và xử lý từng file TSV ────────────────────────────
    mirna_dir = os.path.join(input_dir, "mirna")
    file_paths = glob.glob(os.path.join(mirna_dir, "*.tsv"))

    if not file_paths:
        raise FileNotFoundError(
            f"Không tìm thấy file TSV nào trong: {mirna_dir}\n"
            f"Cấu trúc cần có: input_dir/mirna/*.tsv"
        )

    print(f"\n[miRNA] Tìm thấy {len(file_paths)} file TSV:")
    df_list = []
    for fp in sorted(file_paths):
        df_cancer = clean_and_transpose(fp)
        if not df_cancer.empty:
            df_list.append(df_cancer)

    if not df_list:
        raise ValueError("Tất cả file đều rỗng sau khi xử lý. Kiểm tra lại định dạng TSV.")

    # ── Bước 2: Gộp 4 cancer types ─────────────────────────────────────
    # join='inner': chỉ giữ miRNA có ở tất cả 4 cohorts
    # (đảm bảo ma trận không có NaN do khác nhau về miRNA coverage)
    print(f"\n[miRNA] Gộp {len(df_list)} cancer types (join='inner')...")
    master_df = pd.concat(df_list, axis=0, join='inner')
    print(f"[miRNA] Sau khi gộp: {master_df.shape} (Bệnh nhân × miRNA)")

    # ── Bước 3: Quality Control ─────────────────────────────────────────
    final_df = handle_missing_and_impute(master_df)

    # ── Bước 4: Lọc miRNA biểu hiện thấp ───────────────────────────────
    # Rule không dùng nhãn, tránh leakage: log2(RPM+1) > 0.5 ở >=20% bệnh nhân.
    final_df = filter_low_expression_mirna(final_df)

    # ── Bước 5: Lưu output ──────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "processed_mirna.csv")
    final_df.to_csv(out_path)

    print(f"\n[miRNA] ✓ Đã lưu: {out_path}")
    print(f"[miRNA] ✓ Shape cuối: {final_df.shape}  (Bệnh nhân × miRNA, Xena log2(RPM+1))")

    # Sanity check: phát hiện log transform thiếu/thừa
    max_val = float(final_df.values.max())
    print(f"[miRNA] ✓ Sanity check: max={max_val:.2f}")
    if max_val > 100.0:
        print(f"[miRNA] ⚠️  CẢNH BÁO: max={max_val:.2f} cao bất thường — có thể input là raw RPM, cần log!")
    elif max_val < 6.0:
        print(f"[miRNA] ⚠️  CẢNH BÁO: max={max_val:.2f} thấp bất thường — có thể đã bị double-log!")
    print("="*60)

    return final_df

