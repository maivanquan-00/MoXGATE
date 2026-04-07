"""
preprocess_miRNA.py
===================
Xử lý dữ liệu miRNA Expression từ TCGA cho GIAC dataset.

Hiểu đúng vấn đề:
    - File Xena GDC (mirna.tsv) dùng miRBase v21 → ~1,881 hsa stem-loop features
    - Paper MoXGATE dùng legacy pipeline với miRBase v20 → ~746 hsa stem-loop
    - Chênh lệch 1,881 vs 746 hoàn toàn do VERSION miRBase, không phải mature vs precursor

Giải pháp đúng nhất:
    Tải file hairpin.fa của miRBase v20 → lấy danh sách tên hsa stem-loop v20
    → dùng làm WHITELIST lọc index của file TSV hiện tại
    → Hoàn toàn trung thực với dữ liệu, không áp đặt ngưỡng hay điều kiện nào thêm

    Số features sau lọc = giao thoa giữa:
        (hsa stem-loop có trong v20) ∩ (hsa stem-loop có trong v21 và được quan sát trong dữ liệu)
    → Kỳ vọng: ~746 features, khớp với paper

Pipeline:
    1. Tải/đọc hairpin.fa miRBase v20 → lấy whitelist tên hsa-mir-*
    2. Đọc từng file TSV miRNA
    3. Lọc index theo whitelist v20
    4. Lọc mẫu khối u nguyên phát (-01)
    5. Rút gọn barcode → 12 ký tự
    6. Loại bỏ trùng lặp
    7. Transpose → (Bệnh nhân x miRNA)
    8. Gộp 4 cancer types → join='inner'
    9. Lọc missing > 40% → Median Imputation
    10. Lưu processed_mirna.csv

Kết quả kỳ vọng: ~(1225, 746)
"""

import os
import re
import glob
import gzip
import argparse
import urllib.request
import pandas as pd


# ─────────────────────────────────────────────
# 1. TẢI VÀ ĐỌC WHITELIST miRBase v20
# ─────────────────────────────────────────────

MIRBASE_V20_HAIRPIN_URL = "https://mirbase.org/ftp/20/hairpin.fa.gz"
MIRBASE_V20_HAIRPIN_FALLBACK = "https://github.com/antass/mirbase-archive/raw/main/v20/hairpin.fa.gz"


def download_mirbase_v20_hairpin(save_path: str) -> bool:
    """
    Tải file hairpin.fa.gz của miRBase v20 về đường dẫn save_path.

    Returns:
        True nếu tải thành công, False nếu thất bại.
    """
    for url in [MIRBASE_V20_HAIRPIN_URL, MIRBASE_V20_HAIRPIN_FALLBACK]:
        try:
            print(f"[miRNA] Đang tải miRBase v20 hairpin từ:\n         {url}")
            urllib.request.urlretrieve(url, save_path)
            print(f"[miRNA] ✓ Tải xong → {save_path}")
            return True
        except Exception as e:
            print(f"[miRNA] ✗ Thất bại: {e}")
    return False


def parse_hsa_stemloop_names_from_fasta(fasta_path: str) -> set:
    """
    Đọc file hairpin.fa (plain hoặc .gz) của miRBase,
    trả về set tên stem-loop của HUMAN (hsa-mir-*).

    Định dạng header FASTA miRBase:
        >hsa-let-7a-1 MI0000060 Homo sapiens let-7a-1 stem-loop
        >hsa-mir-21 MI0000077 Homo sapiens miR-21 stem-loop
        ...

    Args:
        fasta_path: Đường dẫn file hairpin.fa hoặc hairpin.fa.gz

    Returns:
        Set các tên stem-loop human, vd: {'hsa-let-7a-1', 'hsa-mir-21', ...}
    """
    hsa_names = set()
    open_func = gzip.open if fasta_path.endswith('.gz') else open

    with open_func(fasta_path, 'rt') as f:
        for line in f:
            if line.startswith('>'):
                # Header dạng: >hsa-let-7a-1 MI0000060 Homo sapiens ...
                name = line[1:].split()[0]  # Lấy phần đầu tiên sau '>'
                if name.startswith('hsa-'):
                    hsa_names.add(name)

    print(f"[miRNA] ✓ Đọc được {len(hsa_names):,} human stem-loop từ miRBase v20")
    return hsa_names


def get_mirbase_v20_whitelist(mirbase_v20_path: str) -> set:
    """
    Lấy whitelist tên hsa stem-loop từ miRBase v20.
    Tự động tải nếu file chưa tồn tại.

    Args:
        mirbase_v20_path: Đường dẫn lưu/đọc file hairpin.fa.gz v20.

    Returns:
        Set tên hsa stem-loop miRBase v20.
    """
    # Tải file nếu chưa có
    if not os.path.exists(mirbase_v20_path):
        os.makedirs(os.path.dirname(mirbase_v20_path), exist_ok=True)
        success = download_mirbase_v20_hairpin(mirbase_v20_path)
        if not success:
            raise RuntimeError(
                "Không thể tải miRBase v20 hairpin.fa.gz tự động.\n"
                "Hãy tải thủ công từ: https://mirbase.org/ftp/20/hairpin.fa.gz\n"
                f"Sau đó đặt vào: {mirbase_v20_path}\n"
                "Rồi chạy lại script."
            )
    else:
        print(f"[miRNA] Dùng miRBase v20 đã có sẵn: {mirbase_v20_path}")

    return parse_hsa_stemloop_names_from_fasta(mirbase_v20_path)


# ─────────────────────────────────────────────
# 2. ĐỌC VÀ XỬ LÝ TỪNG FILE TSV
# ─────────────────────────────────────────────

def clean_and_transpose(file_path: str, v20_whitelist: set) -> pd.DataFrame:
    """
    Đọc 1 file TSV miRNA expression (dạng matrix):
        - Rows: miRNA ID (hsa-mir-*, hsa-let-7*, ...)
        - Cols: TCGA barcodes đầy đủ (TCGA-AA-3819-01A)

    Thực hiện:
        1. Lọc index theo whitelist miRBase v20 (giải pháp cốt lõi)
        2. Lọc mẫu khối u nguyên phát (-01)
        3. Rút gọn barcode → 12 ký tự Patient ID
        4. Loại bỏ trùng lặp (giữ Vial A - first)
        5. Transpose → (Bệnh nhân x miRNA)

    Args:
        file_path:    Đường dẫn file TSV.
        v20_whitelist: Set tên stem-loop hsa từ miRBase v20.

    Returns:
        DataFrame shape (n_patients, n_mirna_v20).
    """
    print(f"  -> Đọc: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path, sep='\t', index_col=0, engine='c')
    print(f"     Shape gốc: {df.shape[0]:,} features × {df.shape[1]:,} samples")

    # --- Bước 1: Lọc theo whitelist miRBase v20 ---
    # Đây là bước cốt lõi: loại bỏ các miRNA mới được thêm vào v21
    # mà không có trong v20 (legacy pipeline của paper)
    n_before = len(df)
    df = df[df.index.isin(v20_whitelist)]
    print(f"     Lọc whitelist v20: {n_before:,} → {len(df):,} features")

    if df.empty:
        print(f"     ⚠ Không còn feature nào sau khi lọc whitelist!")
        print(f"     Kiểm tra 5 index đầu của file: {list(df.index[:5])}")
        return pd.DataFrame()

    # --- Bước 2: Lọc mẫu khối u nguyên phát (-01) ---
    # Barcode trong file Xena đã đầy đủ: TCGA-AA-3819-01A
    # → vị trí [13:15] == '01'
    tumor_cols = [
        col for col in df.columns
        if isinstance(col, str) and len(col) >= 15 and col[13:15] == '01'
    ]
    df_tumor = df[tumor_cols]

    if df_tumor.empty:
        print(f"     ⚠ Không tìm thấy cột tumor (-01)!")
        print(f"     3 cột đầu: {list(df.columns[:3])}")
        return pd.DataFrame()

    # --- Bước 3: Rút gọn barcode → 12 ký tự Patient ID ---
    # TCGA-AA-3819-01A → TCGA-AA-3819
    df_tumor.columns = [col[:12] for col in df_tumor.columns]

    # --- Bước 4: Loại bỏ trùng lặp (giữ Vial A - first) ---
    df_tumor = df_tumor.loc[:, ~df_tumor.columns.duplicated(keep='first')]
    print(f"     Bệnh nhân sau lọc: {df_tumor.shape[1]}")

    # --- Bước 5: Transpose → (Bệnh nhân x miRNA) ---
    return df_tumor.T


# ─────────────────────────────────────────────
# 3. XỬ LÝ MISSING VALUES
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
# 4. HÀM CHÍNH: PROCESS_MIRNA
# ─────────────────────────────────────────────

def process_mirna(
    input_dir: str,
    output_dir: str,
    mirbase_v20_path: str = None,
) -> pd.DataFrame:
    """
    Toàn bộ pipeline xử lý miRNA Expression.

    Args:
        input_dir:        Thư mục gốc omics (chứa subfolder 'mirna/')
        output_dir:       Thư mục lưu output.
        mirbase_v20_path: Đường dẫn file hairpin.fa.gz của miRBase v20.
                          Nếu None → tự động đặt vào output_dir/mirbase_v20_hairpin.fa.gz
                          và tải về nếu chưa có.

    Returns:
        DataFrame đã xử lý, đồng thời lưu ra CSV.
    """
    print("\n" + "="*60)
    print("  BẮT ĐẦU XỬ LÝ miRNA EXPRESSION")
    print("  Phương pháp: Whitelist miRBase v20 (đúng version legacy TCGA)")
    print("="*60)

    # Xác định đường dẫn file miRBase v20
    if mirbase_v20_path is None:
        os.makedirs(output_dir, exist_ok=True)
        mirbase_v20_path = os.path.join(output_dir, "mirbase_v20_hairpin.fa.gz")

    # Bước 1: Lấy whitelist từ miRBase v20
    v20_whitelist = get_mirbase_v20_whitelist(mirbase_v20_path)

    # Bước 2: Đọc và xử lý từng file TSV
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
        df_cancer = clean_and_transpose(fp, v20_whitelist)
        if not df_cancer.empty:
            df_list.append(df_cancer)

    if not df_list:
        raise ValueError(
            "Tất cả file đều rỗng sau khi lọc whitelist.\n"
            "Kiểm tra lại format index trong file TSV (phải dạng hsa-mir-* hoặc hsa-let-7*)."
        )

    # Bước 3: Gộp 4 cancer types → chỉ giữ miRNA có ở TẤT CẢ cancer types
    print(f"\n[miRNA] Gộp {len(df_list)} cancer types (join='inner')...")
    master_df = pd.concat(df_list, axis=0, join='inner')
    print(f"[miRNA] Sau khi gộp: {master_df.shape} (Bệnh nhân x miRNA)")

    # Bước 4: Xử lý missing values
    final_df = handle_missing_and_impute(master_df)

    # Bước 5: Lưu output
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "processed_mirna.csv")
    final_df.to_csv(out_path)

    print(f"\n[miRNA] ✓ Đã lưu: {out_path}")
    print(f"[miRNA] ✓ Shape cuối: {final_df.shape}  (kỳ vọng paper: ~1225 x 746)")

    n_features = final_df.shape[1]
    if 700 <= n_features <= 800:
        print(f"[miRNA] ✓ {n_features} features — khớp tốt với paper (746)!")
    else:
        print(f"[miRNA] ⚠ {n_features} features — lệch so với paper (746).")
        print(f"         Có thể do tên miRNA đổi giữa v20 và v21 (rename/merge).")
        print(f"         Đây là giới hạn của việc dùng dữ liệu GDC harmonized thay vì legacy.")

    print("="*60)
    return final_df


# ─────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Xử lý miRNA Expression TCGA cho GIAC dataset (whitelist miRBase v20)"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Thư mục gốc chứa dữ liệu omics (chứa subfolder 'mirna/')"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Thư mục lưu file processed_mirna.csv"
    )
    parser.add_argument(
        "--mirbase_v20_path", type=str, default=None,
        help="(Tùy chọn) Đường dẫn file hairpin.fa.gz của miRBase v20. "
             "Nếu không cung cấp, script sẽ tự động tải về output_dir/mirbase_v20_hairpin.fa.gz."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_mirna(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mirbase_v20_path=args.mirbase_v20_path,
    )