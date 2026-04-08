"""
preprocess_miRNA_zero.py
========================
Biến thể thử nghiệm của preprocess_miRNA.py — có thêm bước lọc
miRNA không có biểu hiện ở 100% bệnh nhân (zero-expression filter).

Nguồn dữ liệu:
    GDC TCGA Harmonized — miRNA Expression Quantification (stem-loop, miRBase v22)

Pipeline:
    ┌─────────────────────────────────────────────────────────────────┐
    │ BƯỚC 1 — Xử lý từng cancer type (lặp qua 4 file TSV)            │
    │   1a. Lọc mẫu khối u nguyên phát: barcode[13:15] == '01'        │
    │   1b. Rút gọn barcode → 12 ký tự Patient ID                     │
    │   1c. Loại bỏ bệnh nhân trùng lặp (giữ Vial A = first)          │
    │   1d. Transpose → (Bệnh nhân × miRNA)                           │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 2 — Gộp 4 cancer types                                     │
    │   pd.concat(join='inner') → chỉ giữ miRNA có ở tất cả cohorts   │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 3 — Zero-expression filter  KHÁC SO VỚI preprocess_miRNA   │
    │   Loại miRNA có giá trị = 0 ở 100% bệnh nhân trên toàn cohort   │
    │   Cơ sở: node feature toàn 0 → không đóng góp được cho model    │
    │   Không áp dụng ngưỡng 90% để tránh loại miRNA có expression    │
    │   thưa thớt nhưng vẫn có ý nghĩa sinh học.                      │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 4 — Quality Control                                        │
    │   Loại miRNA thiếu dữ liệu ở >40% bệnh nhân                     │
    │   Điền phần còn lại bằng Median Imputation                      │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 5 — Lưu kết quả                                            │
    │   → processed_mirna_zero.csv  (Bệnh nhân × miRNA)               │
    └─────────────────────────────────────────────────────────────────┘

So sánh với preprocess_miRNA.py (pipeline gốc không filter):
    - preprocess_miRNA.py      → ~1,881 features (toàn bộ GDC)
    - preprocess_miRNA_zero.py → ~1,631 features (sau khi bỏ zero-100%)

Lưu ý:
    - File này dùng để so sánh / thử nghiệm, không phải pipeline chính.
    - Kết quả được lưu vào processed_mirna_zero.csv (không ghi đè file chính).
"""

import os
import glob
import argparse
import pandas as pd


# ─────────────────────────────────────────────
# 1. ĐỌC VÀ XỬ LÝ TỪNG FILE TSV
# ─────────────────────────────────────────────

def clean_and_transpose(file_path: str) -> pd.DataFrame:
    """
    Đọc 1 file TSV miRNA expression (GDC dạng matrix):
        - Rows: miRNA stem-loop ID (hsa-mir-21, hsa-let-7a-1, ...)
        - Cols: TCGA barcodes đầy đủ (TCGA-AA-3819-01A-...)
        - Values: read_count hoặc RPM

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
    tumor_cols = [
        col for col in df.columns
        if isinstance(col, str) and len(col) >= 15 and col[13:15] == '01'
    ]
    df_tumor = df[tumor_cols]

    if df_tumor.empty:
        print(f"     ⚠ Không tìm thấy cột tumor (-01)! 3 cột đầu: {list(df.columns[:3])}")
        return pd.DataFrame()

    # --- Bước 1b: Rút gọn barcode → 12 ký tự Patient ID ---
    df_tumor.columns = [col[:12] for col in df_tumor.columns]

    # --- Bước 1c: Loại bỏ bệnh nhân trùng lặp (giữ Vial A = first) ---
    df_tumor = df_tumor.loc[:, ~df_tumor.columns.duplicated(keep='first')]
    print(f"     Bệnh nhân sau lọc: {df_tumor.shape[1]}")

    # --- Bước 1d: Transpose → (Bệnh nhân × miRNA) ---
    return df_tumor.T


# ─────────────────────────────────────────────
# 2. ZERO-EXPRESSION FILTER
# ─────────────────────────────────────────────

def filter_zero_expression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 3 — Loại miRNA có giá trị = 0 ở 100% bệnh nhân trên toàn cohort.

    Cơ sở khoa học:
        - miRNA không có bất kỳ biểu hiện nào ở toàn bộ cohort không mang
          thông tin phân biệt giữa các bệnh nhân.
        - Trong graph-based model: node feature vector toàn 0 → không đóng
          góp được qua bất kỳ phép nhân ma trận nào trong GAT/GCN.
        - Không dùng ngưỡng 90% để tránh loại miRNA có expression thưa thớt
          nhưng vẫn có ý nghĩa sinh học với một số bệnh nhân cụ thể.

    Args:
        df: DataFrame (n_patients, n_mirna) — ma trận tổng đã gộp.

    Returns:
        DataFrame sau khi loại các cột toàn 0.
    """
    n_before = df.shape[1]

    # Kiểm tra từng cột (miRNA): True nếu TẤT CẢ bệnh nhân đều = 0
    zero_mask = (df == 0).all(axis=0)
    n_zero = zero_mask.sum()

    df_filtered = df.loc[:, ~zero_mask]

    print(f"[miRNA-zero] Bước 3 — Zero-expression filter:")
    print(f"             Trước: {n_before:,} miRNA")
    print(f"             Loại:  {n_zero:,} miRNA (zero ở 100% bệnh nhân)")
    print(f"             Sau:   {df_filtered.shape[1]:,} miRNA")

    return df_filtered


# ─────────────────────────────────────────────
# 3. XỬ LÝ MISSING VALUES
# ─────────────────────────────────────────────

def handle_missing_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 4 — Quality Control trên ma trận tổng:
        4a. Loại miRNA thiếu dữ liệu ở >40% bệnh nhân
        4b. Median Imputation cho phần còn lại

    Args:
        df: DataFrame (n_patients, n_mirna).

    Returns:
        DataFrame đã lọc và impute.
    """
    print(f"\n[miRNA-zero] Kích thước trước QC: {df.shape} (Bệnh nhân × miRNA)")

    # 4a: miRNA hợp lệ nếu có giá trị ở ít nhất 60% bệnh nhân
    min_valid_count = int(len(df) * 0.60)
    df_filtered = df.dropna(axis=1, thresh=min_valid_count)
    n_dropped = df.shape[1] - df_filtered.shape[1]
    if n_dropped > 0:
        print(f"[miRNA-zero]  Loại {n_dropped:,} miRNA thiếu >40% → còn {df_filtered.shape[1]:,} miRNA")

    # 4b: Điền khuyết bằng Median của từng miRNA
    df_imputed = df_filtered.fillna(df_filtered.median())

    print(f"[miRNA-zero] Kích thước sau QC: {df_imputed.shape}")
    return df_imputed


# ─────────────────────────────────────────────
# 4. HÀM CHÍNH: PROCESS_MIRNA_ZERO
# ─────────────────────────────────────────────

def process_mirna_zero(input_dir: str, output_dir: str) -> pd.DataFrame:
    """
    Pipeline xử lý miRNA Expression với zero-expression filter.

    Args:
        input_dir:  Thư mục gốc omics (chứa subfolder 'mirna/' bên trong)
        output_dir: Thư mục lưu output.

    Returns:
        DataFrame (Bệnh nhân × miRNA), đồng thời lưu ra CSV.
    """
    print("\n" + "="*60)
    print("  BẮT ĐẦU XỬ LÝ miRNA EXPRESSION (+ Zero Filter)")
    print("  Nguồn: GDC TCGA Harmonized (miRBase v22)")
    print("="*60)

    # ── Bước 1: Đọc và xử lý từng file TSV ────────────────────────────
    mirna_dir = os.path.join(input_dir, "mirna")
    file_paths = glob.glob(os.path.join(mirna_dir, "*.tsv"))

    if not file_paths:
        raise FileNotFoundError(
            f"Không tìm thấy file TSV nào trong: {mirna_dir}\n"
            f"Cấu trúc cần có: input_dir/mirna/*.tsv"
        )

    print(f"\n[miRNA-zero] Tìm thấy {len(file_paths)} file TSV:")
    df_list = []
    for fp in sorted(file_paths):
        df_cancer = clean_and_transpose(fp)
        if not df_cancer.empty:
            df_list.append(df_cancer)

    if not df_list:
        raise ValueError("Tất cả file đều rỗng sau khi xử lý. Kiểm tra lại định dạng TSV.")

    # ── Bước 2: Gộp 4 cancer types ─────────────────────────────────────
    print(f"\n[miRNA-zero] Gộp {len(df_list)} cancer types (join='inner')...")
    master_df = pd.concat(df_list, axis=0, join='inner')
    print(f"[miRNA-zero] Sau khi gộp: {master_df.shape} (Bệnh nhân × miRNA)")

    # ── Bước 3: Zero-expression filter ─────────────────────────────────
    # Loại miRNA zero ở 100% bệnh nhân TRÊN TOÀN COHORT (sau khi gộp)
    # Phải làm sau khi gộp để đánh giá trên toàn bộ 1,225 bệnh nhân
    master_df = filter_zero_expression(master_df)

    # ── Bước 4: Quality Control ─────────────────────────────────────────
    final_df = handle_missing_and_impute(master_df)

    # ── Bước 5: Lưu output ──────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "processed_mirna_zero.csv")
    final_df.to_csv(out_path)

    print(f"\n[miRNA-zero] ✓ Đã lưu: {out_path}")
    print(f"[miRNA-zero] ✓ Shape cuối: {final_df.shape}  (Bệnh nhân × miRNA, sau zero filter)")
    print("="*60)

    return final_df


# ─────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Xử lý miRNA Expression TCGA GDC với zero-expression filter"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Thư mục gốc chứa dữ liệu omics (chứa subfolder 'mirna/')"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Thư mục lưu file processed_mirna_zero.csv"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_mirna_zero(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
