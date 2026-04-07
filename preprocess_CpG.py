"""
preprocess_CpG.py
=================
Xử lý dữ liệu DNA Methylation (CpG) từ TCGA cho GIAC dataset.

Đặc thù của Methylation:
    - Có 2 loại chip: Illumina 27k và 450k
    - 27k chứa ~27,578 CpG sites (tập con của 450k)
    - 450k chứa ~485,512 CpG sites
    - join='inner' khi gộp sẽ tự động lấy giao thoa → chỉ giữ ~27k sites chung
    - Paper dùng 23,381 features → sau khi lọc cross-reactive probes & sex chr

Pipeline:
    1. Đọc tất cả file TSV (cả 27k và 450k từ subfolder)
    2. Lọc mẫu khối u nguyên phát (-01)
    3. Rút gọn barcode → 12 ký tự
    4. Loại bỏ trùng lặp
    5. Transpose → (Bệnh nhân x CpG)
    6. Gộp tất cả → join='inner' (tự thu về ~27k sites chung)
    7. Lọc cross-reactive probes (Chen et al. 2013)
    8. Lọc probes trên nhiễm sắc thể giới tính (chrX, chrY)
       bằng Illumina 450k Manifest
    9. Lọc missing > 40% → Median Imputation
    10. Lưu processed_methylation.csv

Cấu trúc thư mục cần có:
    input_dir/methyl/
    ├── 27k/
    │   ├── TCGA-COAD.methylation27.tsv
    │   ├── TCGA-READ.methylation27.tsv
    │   └── TCGA-STAD.methylation27.tsv
    └── 450k/
        ├── TCGA-COAD.methylation450.tsv
        ├── TCGA-ESCA.methylation450.tsv
        ├── TCGA-READ.methylation450.tsv
        └── TCGA-STAD.methylation450.tsv

Kết quả kỳ vọng: ~(1255, 23381) — khớp với paper MoXGATE

Các file phụ trợ cần có (tùy chọn, để lọc chính xác hơn):
    - cross_reactive_probes.txt : Danh sách probe cross-reactive (Chen et al. 2013)
      Tải từ: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL16304
    - HumanMethylation450_manifest.csv : Illumina 450k manifest
      Tải từ: https://support.illumina.com/downloads/infinium-methylation-450k-product-files.html
    Nếu không có → script vẫn chạy được, chỉ bỏ qua bước lọc tương ứng.
"""

import os
import glob
import argparse
import pandas as pd


# ─────────────────────────────────────────────
# 1. TẢI DANH SÁCH PROBES CẦN LỌC (TÙY CHỌN)
# ─────────────────────────────────────────────

def load_cross_reactive_probes(cross_reactive_path: str) -> set:
    """
    Đọc danh sách cross-reactive probes (Chen et al. 2013).
    File thường là 1 cột probe ID, có thể có header.

    Args:
        cross_reactive_path: Đường dẫn file txt/csv chứa probe IDs.

    Returns:
        Set các probe ID cần loại bỏ.
    """
    if not cross_reactive_path or not os.path.exists(cross_reactive_path):
        print("[CpG]  ⚠ Không tìm thấy file cross-reactive probes → bỏ qua bước lọc này")
        return set()

    df = pd.read_csv(cross_reactive_path, header=None, comment='#')
    probes = set(df.iloc[:, 0].astype(str).str.strip().tolist())
    print(f"[CpG]  ✓ Đọc {len(probes):,} cross-reactive probes")
    return probes


def load_sex_chromosome_probes(manifest_path: str) -> set:
    """
    Đọc Illumina 450k manifest, trả về probes trên chrX và chrY.
    Manifest có thể có vài dòng header (skiprows=7 cho file Illumina gốc).

    Args:
        manifest_path: Đường dẫn file manifest CSV.

    Returns:
        Set probe ID trên nhiễm sắc thể giới tính.
    """
    if not manifest_path or not os.path.exists(manifest_path):
        print("[CpG]  ⚠ Không tìm thấy Illumina manifest → bỏ qua lọc sex chromosomes")
        return set()

    # Thử đọc với skiprows=7 (format Illumina gốc)
    # Nếu file đã được làm sạch thì skiprows=0
    try:
        manifest = pd.read_csv(
            manifest_path, skiprows=7, index_col=0,
            low_memory=False, usecols=lambda c: c in ['Name', 'CHR'] or c == 0
        )
    except Exception:
        manifest = pd.read_csv(
            manifest_path, index_col=0,
            low_memory=False
        )

    if 'CHR' not in manifest.columns:
        print("[CpG]  ⚠ Không tìm thấy cột 'CHR' trong manifest → bỏ qua lọc sex chromosomes")
        return set()

    sex_probes = set(manifest[manifest['CHR'].isin(['X', 'Y'])].index.astype(str))
    print(f"[CpG]  ✓ Tìm thấy {len(sex_probes):,} probes trên chrX/Y")
    return sex_probes


# ─────────────────────────────────────────────
# 2. ĐỌC VÀ XỬ LÝ TỪNG FILE TSV
# ─────────────────────────────────────────────

def clean_and_transpose(file_path: str) -> pd.DataFrame:
    """
    Đọc 1 file TSV methylation:
        - Rows: CpG probe ID (vd: cg00000029)
        - Cols: TCGA barcodes

    Thực hiện:
        1. Lọc mẫu khối u nguyên phát (-01)
        2. Rút gọn barcode → 12 ký tự
        3. Loại bỏ trùng lặp
        4. Transpose → (Bệnh nhân x CpG)

    Args:
        file_path: Đường dẫn file TSV.

    Returns:
        DataFrame shape (n_patients, n_cpg).
    """
    print(f"  -> Đọc: {os.path.basename(file_path)}")

    df = pd.read_csv(file_path, sep='\t', index_col=0, engine='c')

    # --- Lọc mẫu khối u nguyên phát (-01) ---
    tumor_cols = [
        col for col in df.columns
        if isinstance(col, str) and len(col) >= 15 and col[13:15] == '01'
    ]
    df_tumor = df[tumor_cols]

    # --- Rút gọn barcode → 12 ký tự ---
    df_tumor.columns = [col[:12] for col in df_tumor.columns]

    # --- Loại bỏ trùng lặp (giữ Vial A) ---
    df_tumor = df_tumor.loc[:, ~df_tumor.columns.duplicated(keep='first')]

    print(f"     CpG sites: {df_tumor.shape[0]:,} | Bệnh nhân: {df_tumor.shape[1]}")

    return df_tumor.T


# ─────────────────────────────────────────────
# 3. LỌC PROBES KHÔNG ĐÁNG TIN CẬY
# ─────────────────────────────────────────────

def filter_unreliable_probes(
    df: pd.DataFrame,
    cross_reactive_probes: set,
    sex_chromosome_probes: set
) -> pd.DataFrame:
    """
    Loại bỏ các CpG probes không đáng tin cậy:
        1. Cross-reactive probes (bám vào nhiều vị trí → nhiễu)
        2. Probes trên chrX/Y (bias giới tính)

    Args:
        df:                      DataFrame (n_patients, n_cpg).
        cross_reactive_probes:   Set probe IDs cross-reactive.
        sex_chromosome_probes:   Set probe IDs trên chrX/Y.

    Returns:
        DataFrame sau khi lọc probes xấu.
    """
    n_before = df.shape[1]
    bad_probes = cross_reactive_probes | sex_chromosome_probes

    if bad_probes:
        keep = [col for col in df.columns if col not in bad_probes]
        df = df[keep]
        print(f"[CpG]  Lọc probes không tin cậy: {n_before:,} → {df.shape[1]:,} "
              f"(bỏ {n_before - df.shape[1]:,} probes)")
    else:
        print("[CpG]  Bỏ qua bước lọc probes (không có file tham chiếu)")

    return df


# ─────────────────────────────────────────────
# 4. XỬ LÝ MISSING VALUES
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
    print(f"\n[CpG]  Kích thước trước lọc missing: {df.shape}")

    min_valid_count = int(len(df) * 0.60)
    df_filtered = df.dropna(axis=1, thresh=min_valid_count)
    df_imputed = df_filtered.fillna(df_filtered.median())

    print(f"[CpG]  Kích thước sau lọc & impute:  {df_imputed.shape}")
    return df_imputed


# ─────────────────────────────────────────────
# 5. HÀM CHÍNH: PROCESS_CPG
# ─────────────────────────────────────────────

def process_cpg(
    input_dir: str,
    output_dir: str,
    cross_reactive_path: str = None,
    manifest_path: str = None,
) -> pd.DataFrame:
    """
    Toàn bộ pipeline xử lý DNA Methylation.

    Args:
        input_dir:             Thư mục gốc omics (chứa subfolder 'methyl/27k/' và 'methyl/450k/')
        output_dir:            Thư mục lưu output.
        cross_reactive_path:   (Tùy chọn) Đường dẫn file cross-reactive probes.
        manifest_path:         (Tùy chọn) Đường dẫn Illumina 450k manifest.

    Returns:
        DataFrame đã xử lý, đồng thời lưu ra CSV.
    """
    print("\n" + "="*60)
    print("  BẮT ĐẦU XỬ LÝ DNA METHYLATION (CpG)")
    print("="*60)

    # Thu thập tất cả file TSV từ cả 2 subfolder
    methyl_dir = os.path.join(input_dir, "methyl")
    path_27k  = glob.glob(os.path.join(methyl_dir, "27k",  "*.tsv"))
    path_450k = glob.glob(os.path.join(methyl_dir, "450k", "*.tsv"))
    file_paths = path_27k + path_450k

    if not file_paths:
        # Fallback: thử đọc thẳng trong methyl/ (không có subfolder)
        file_paths = glob.glob(os.path.join(methyl_dir, "*.tsv"))

    if not file_paths:
        raise FileNotFoundError(
            f"Không tìm thấy file TSV nào trong: {methyl_dir}\n"
            f"Cấu trúc cần có: input_dir/methyl/27k/*.tsv và input_dir/methyl/450k/*.tsv"
        )

    print(f"\n[CpG]  Tìm thấy {len(path_27k)} file 27k + {len(path_450k)} file 450k "
          f"= {len(file_paths)} file tổng cộng")

    # Đọc và xử lý từng file
    df_list = []
    for fp in sorted(file_paths):
        df_cancer = clean_and_transpose(fp)
        df_list.append(df_cancer)

    # Gộp tất cả — join='inner' tự động lấy giao thoa CpG sites
    # (thu 450k về ~27k sites chung)
    print(f"\n[CpG]  Gộp {len(df_list)} file (join='inner' → lấy CpG sites chung)...")
    master_df = pd.concat(df_list, axis=0, join='inner')
    print(f"[CpG]  Sau khi gộp: {master_df.shape} (Bệnh nhân x CpG sites)")

    # Lọc cross-reactive probes và sex chromosome probes
    cross_reactive = load_cross_reactive_probes(cross_reactive_path)
    sex_chr_probes = load_sex_chromosome_probes(manifest_path)
    master_df = filter_unreliable_probes(master_df, cross_reactive, sex_chr_probes)

    # Xử lý missing values
    final_df = handle_missing_and_impute(master_df)

    # Lưu output
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "processed_methylation.csv")
    final_df.to_csv(out_path)
    print(f"\n[CpG]  ✓ Đã lưu: {out_path}")
    print(f"[CpG]  ✓ Shape cuối: {final_df.shape}  (kỳ vọng: ~1255 x 23381)")
    print("="*60)

    return final_df


# ─────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Xử lý DNA Methylation TCGA cho GIAC dataset"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Thư mục gốc chứa dữ liệu omics (chứa subfolder 'methyl/27k/' và 'methyl/450k/')"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Thư mục lưu file processed_methylation.csv"
    )
    parser.add_argument(
        "--cross_reactive_path", type=str, default=None,
        help="(Tùy chọn) Đường dẫn file cross-reactive probes (Chen et al. 2013)"
    )
    parser.add_argument(
        "--manifest_path", type=str, default=None,
        help="(Tùy chọn) Đường dẫn Illumina 450k manifest CSV"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_cpg(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cross_reactive_path=args.cross_reactive_path,
        manifest_path=args.manifest_path,
    )