"""
preprocess_miRNA.py
===================
Xử lý dữ liệu miRNA Expression từ TCGA cho GIAC dataset.

Tại sao file Xena GDC có 1,881 features thay vì 746 như paper?
    - File Xena GDC dùng miRBase v21 → ~1,881 hsa stem-loop
    - Paper MoXGATE dùng legacy pipeline với miRBase v20 → ~746 hsa stem-loop
    - Nguyên nhân: chênh lệch VERSION miRBase, không phải mature vs precursor

Giải pháp đúng nhất — Whitelist miRBase v20:
    Dùng file hairpin.fa của miRBase v20 (đã có sẵn trên Drive) để lấy danh sách
    tên hsa stem-loop → dùng làm whitelist lọc index của file TSV GDC v21
    → Không áp đặt bất kỳ ngưỡng hay điều kiện nào thêm, hoàn toàn trung thực

Cấu trúc file hairpin.fa miRBase:
    >cel-let-7 MI0000001 Caenorhabditis elegans let-7 stem-loop
    UACACUGUGGAUCCGGUGAGGUAGUAGG...
    >hsa-let-7a-1 MI0000060 Homo sapiens let-7a-1 stem-loop
    UGGGAUGAGGUAGUAGGUUGUAUAGUU...

    → Lấy phần đầu của header (sau '>') = tên stem-loop
    → Chỉ giữ các tên bắt đầu bằng 'hsa-' (human)

Pipeline:
    1. Đọc hairpin.fa v20 → whitelist ~946 hsa stem-loop
    2. Đọc từng file TSV → lọc index theo whitelist
    3. Lọc mẫu khối u nguyên phát (-01)
    4. Rút gọn barcode → 12 ký tự
    5. Loại bỏ trùng lặp
    6. Transpose → (Bệnh nhân x miRNA)
    7. Gộp 4 cancer types → join='inner'
    8. Lọc missing > 40% → Median Imputation
    9. Lưu processed_mirna.csv

File cần có sẵn:
    /content/drive/MyDrive/ĐATN_2025.2/data_original/annotation/hairpin.fa
    (Tải từ https://mirbase.org/ftp/20/hairpin.fa.gz rồi giải nén)

Kết quả kỳ vọng: ~(1225, 746)
"""

import os
import gzip
import glob
import argparse
import pandas as pd


# Đường dẫn mặc định tới file hairpin.fa miRBase v20 trên Google Drive
DEFAULT_MIRBASE_V20_PATH = (
    "/content/drive/MyDrive/ĐATN_2025.2/data_original/annotation/hairpin.fa"
)


# ─────────────────────────────────────────────
# 1. ĐỌC WHITELIST TỪ HAIRPIN.FA miRBase v20
# ─────────────────────────────────────────────

def parse_hsa_stemloop_names(fasta_path: str) -> set:
    """
    Đọc file hairpin.fa (plain hoặc .gz) của miRBase,
    trả về set tên stem-loop của HUMAN (prefix 'hsa-').

    Định dạng header FASTA miRBase:
        >hsa-let-7a-1 MI0000060 Homo sapiens let-7a-1 stem-loop
        >cel-let-7 MI0000001 Caenorhabditis elegans let-7 stem-loop

    → Lấy token đầu tiên sau '>' → kiểm tra startswith('hsa-')

    Args:
        fasta_path: Đường dẫn file hairpin.fa hoặc hairpin.fa.gz

    Returns:
        Set các tên stem-loop human, vd: {'hsa-let-7a-1', 'hsa-mir-21', ...}
    """
    hsa_names = set()
    open_func = gzip.open if fasta_path.endswith('.gz') else open

    with open_func(fasta_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('>'):
                # Lấy tên: token đầu tiên sau '>'
                # Ví dụ: '>hsa-let-7a-1 MI0000060 ...' → 'hsa-let-7a-1'
                name = line[1:].split()[0]
                if name.startswith('hsa-'):
                    hsa_names.add(name)

    print(f"[miRNA] ✓ Whitelist miRBase v20: {len(hsa_names):,} human stem-loop")
    return hsa_names


def get_mirbase_v20_whitelist(mirbase_v20_path: str) -> set:
    """
    Đọc file hairpin.fa của miRBase v20, trả về whitelist tên hsa stem-loop.

    Args:
        mirbase_v20_path: Đường dẫn file hairpin.fa hoặc hairpin.fa.gz của miRBase v20.

    Returns:
        Set tên hsa stem-loop miRBase v20.

    Raises:
        FileNotFoundError: Nếu không tìm thấy file.
    """
    if not os.path.exists(mirbase_v20_path):
        raise FileNotFoundError(
            f"Không tìm thấy file miRBase v20 tại:\n  {mirbase_v20_path}\n\n"
            "Hãy tải file hairpin.fa từ:\n"
            "  https://mirbase.org/ftp/20/hairpin.fa.gz\n"
            "Giải nén rồi đặt vào thư mục annotation trên Drive."
        )

    print(f"[miRNA] Đọc whitelist từ: {os.path.basename(mirbase_v20_path)}")
    return parse_hsa_stemloop_names(mirbase_v20_path)


# ─────────────────────────────────────────────
# 2. ĐỌC VÀ XỬ LÝ TỪNG FILE TSV
# ─────────────────────────────────────────────

def clean_and_transpose(file_path: str, v20_whitelist: set) -> pd.DataFrame:
    """
    Đọc 1 file TSV miRNA expression (dạng matrix):
        - Rows: miRNA ID (hsa-mir-21, hsa-let-7a-1, ...)
        - Cols: TCGA barcodes đầy đủ (TCGA-AA-3819-01A)
        - Values: log2 RPM hoặc RPM (tùy nguồn)

    Thực hiện:
        1. Lọc index theo whitelist miRBase v20  ← bước cốt lõi
        2. Lọc mẫu khối u nguyên phát (-01)
        3. Rút gọn barcode → 12 ký tự Patient ID
        4. Loại bỏ trùng lặp (giữ Vial A - first)
        5. Transpose → (Bệnh nhân x miRNA)

    Args:
        file_path:     Đường dẫn file TSV.
        v20_whitelist: Set tên stem-loop hsa từ miRBase v20.

    Returns:
        DataFrame shape (n_patients, n_mirna_v20).
    """
    print(f"  -> Đọc: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path, sep='\t', index_col=0, engine='c')
    print(f"     Shape gốc: {df.shape[0]:,} features × {df.shape[1]:,} samples")

    # --- Bước 1: Lọc theo whitelist miRBase v20 ---
    # Loại bỏ các stem-loop được thêm mới trong v21 (không có trong v20/legacy)
    n_before = len(df)
    df = df[df.index.isin(v20_whitelist)]
    print(f"     Whitelist v20: {n_before:,} → {len(df):,} features")

    if df.empty:
        print(f"     ⚠ Không còn feature nào sau lọc whitelist!")
        print(f"     5 index đầu trong file: {list(df.index[:5])}")
        return pd.DataFrame()

    # --- Bước 2: Lọc mẫu khối u nguyên phát (-01) ---
    # Barcode Xena đã đầy đủ: TCGA-AA-3819-01A → [13:15] == '01'
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
        mirbase_v20_path: Đường dẫn file hairpin.fa (hoặc .fa.gz) của miRBase v20.
                          Mặc định: DEFAULT_MIRBASE_V20_PATH (trỏ vào Drive)

    Returns:
        DataFrame đã xử lý, đồng thời lưu ra CSV.
    """
    print("\n" + "="*60)
    print("  BẮT ĐẦU XỬ LÝ miRNA EXPRESSION")
    print("  Phương pháp: Whitelist miRBase v20")
    print("="*60)

    # Dùng đường dẫn mặc định nếu không truyền vào
    if mirbase_v20_path is None:
        mirbase_v20_path = DEFAULT_MIRBASE_V20_PATH
        print(f"[miRNA] Dùng đường dẫn mặc định:\n        {mirbase_v20_path}")

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
            "Kiểm tra lại: index trong TSV phải dạng 'hsa-mir-*' hoặc 'hsa-let-7*'."
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
        print(f"         Có thể do một số miRNA đổi tên giữa v20 và v21.")
        print(f"         Đây là giới hạn của việc dùng GDC harmonized data thay vì legacy.")

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
        help=(
            f"Đường dẫn file hairpin.fa (hoặc .fa.gz) của miRBase v20. "
            f"Mặc định: {DEFAULT_MIRBASE_V20_PATH}"
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_mirna(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mirbase_v20_path=args.mirbase_v20_path,
    )