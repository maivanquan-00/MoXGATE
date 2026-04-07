"""
preprocess_miRNA.py
===================
Xử lý dữ liệu miRNA Expression từ TCGA cho GIAC dataset.

Vấn đề với dữ liệu GDC hiện tại:
    - File GDC mới chứa ~1,881 features (bao gồm cả precursor + mature miRNA)
    - Paper MoXGATE dùng 746 features → chỉ mature miRNA (hsa-miR-*)
    - Mature miRNA: tên viết hoa chữ R (hsa-miR-21-5p)
    - Precursor miRNA: tên viết thường chữ r (hsa-mir-21)

Pipeline:
    1. Đọc từng file TSV miRNA
    2. Lọc chỉ giữ mature miRNA (index chứa 'hsa-miR-')
    3. Lọc mẫu khối u nguyên phát (-01)
    4. Rút gọn barcode → 12 ký tự Patient ID
    5. Loại bỏ trùng lặp (giữ Vial A)
    6. Transpose → (Bệnh nhân x miRNA)
    7. Gộp 4 cancer types → join='inner'
    8. Lọc missing > 40% → Median Imputation
    9. Lưu processed_mirna.csv

Kết quả kỳ vọng: ~(1225, 746) — khớp với paper MoXGATE
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
    Đọc 1 file TSV miRNA expression:
        - Rows: miRNA ID (hsa-miR-21-5p hoặc hsa-mir-21)
        - Cols: TCGA barcodes

    Thực hiện:
        1. Lọc chỉ giữ mature miRNA (hsa-miR-* — chữ R hoa)
        2. Lọc mẫu khối u nguyên phát (-01)
        3. Rút gọn barcode → 12 ký tự
        4. Loại bỏ trùng lặp
        5. Transpose → (Bệnh nhân x miRNA)

    Args:
        file_path: Đường dẫn file TSV.

    Returns:
        DataFrame shape (n_patients, n_mature_mirna).
    """
    print(f"  -> Đọc: {os.path.basename(file_path)}")

    df = pd.read_csv(file_path, sep='\t', index_col=0, engine='c')

    # --- Bước 1: Lọc mature miRNA ---
    # Mature: hsa-miR-21-5p  (chữ R HOA)
    # Precursor: hsa-mir-21  (chữ r thường)
    n_before = len(df)
    mature_mask = df.index.str.contains('hsa-miR-', na=False)
    df = df[mature_mask]
    print(f"     Lọc mature miRNA: {n_before:,} → {len(df):,} features")

    # --- Bước 2: Lọc mẫu khối u nguyên phát (-01) ---
    tumor_cols = [
        col for col in df.columns
        if isinstance(col, str) and len(col) >= 15 and col[13:15] == '01'
    ]
    df_tumor = df[tumor_cols]

    # --- Bước 3: Rút gọn barcode → 12 ký tự Patient ID ---
    df_tumor.columns = [col[:12] for col in df_tumor.columns]

    # --- Bước 4: Loại bỏ trùng lặp (giữ Vial A - first) ---
    df_tumor = df_tumor.loc[:, ~df_tumor.columns.duplicated(keep='first')]

    print(f"     Bệnh nhân sau lọc: {df_tumor.shape[1]}")

    # --- Bước 5: Transpose → (Bệnh nhân x miRNA) ---
    return df_tumor.T


# ─────────────────────────────────────────────
# 2. XỬ LÝ MISSING VALUES
# ─────────────────────────────────────────────

def handle_missing_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý missing values:
        1. Loại features thiếu > 40% (thresh = 60% bệnh nhân)
        2. Median Imputation cho phần còn lại

    Args:
        df: DataFrame (n_patients, n_features).

    Returns:
        DataFrame đã lọc và impute.
    """
    print(f"\n[miRNA] Kích thước trước lọc: {df.shape} (Bệnh nhân x Features)")

    min_valid_count = int(len(df) * 0.60)
    df_filtered = df.dropna(axis=1, thresh=min_valid_count)
    df_imputed = df_filtered.fillna(df_filtered.median())

    print(f"[miRNA] Kích thước sau lọc & impute: {df_imputed.shape}")
    return df_imputed


# ─────────────────────────────────────────────
# 3. HÀM CHÍNH: PROCESS_MIRNA
# ─────────────────────────────────────────────

def process_mirna(input_dir: str, output_dir: str) -> pd.DataFrame:
    """
    Toàn bộ pipeline xử lý miRNA Expression.

    Args:
        input_dir:  Thư mục gốc omics (chứa subfolder 'mirna/')
        output_dir: Thư mục lưu output.

    Returns:
        DataFrame đã xử lý, đồng thời lưu ra CSV.
    """
    print("\n" + "="*60)
    print("  BẮT ĐẦU XỬ LÝ miRNA EXPRESSION")
    print("="*60)

    mirna_dir = os.path.join(input_dir, "mirna")
    file_paths = glob.glob(os.path.join(mirna_dir, "*.tsv"))

    if not file_paths:
        raise FileNotFoundError(
            f"Không tìm thấy file TSV nào trong: {mirna_dir}\n"
            f"Kiểm tra lại cấu trúc thư mục: input_dir/mirna/*.tsv"
        )

    print(f"\n[miRNA] Tìm thấy {len(file_paths)} file TSV:")
    df_list = []
    for fp in sorted(file_paths):
        df_cancer = clean_and_transpose(fp)
        df_list.append(df_cancer)

    # Gộp 4 cancer types, lấy giao thoa features
    print(f"\n[miRNA] Gộp {len(df_list)} cancer types (join='inner')...")
    master_df = pd.concat(df_list, axis=0, join='inner')
    print(f"[miRNA] Sau khi gộp: {master_df.shape} (Bệnh nhân x miRNA)")

    # Xử lý missing values
    final_df = handle_missing_and_impute(master_df)

    # Lưu output
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "processed_mirna.csv")
    final_df.to_csv(out_path)
    print(f"\n[miRNA] ✓ Đã lưu: {out_path}")
    print(f"[miRNA] ✓ Shape cuối: {final_df.shape}  (kỳ vọng: ~1225 x 746)")
    print("="*60)

    return final_df


# ─────────────────────────────────────────────
# 4. ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Xử lý miRNA Expression TCGA cho GIAC dataset"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Thư mục gốc chứa dữ liệu omics (chứa subfolder 'mirna/')"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Thư mục lưu file processed_mirna.csv"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_mirna(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )