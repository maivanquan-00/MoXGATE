"""
preprocess_CpG.py
=================
Xử lý dữ liệu DNA Methylation (CpG) từ TCGA GDC cho GIAC dataset.

Nguồn dữ liệu:
    GDC TCGA Harmonized — Illumina Human Methylation 27k và 450k
    (COAD: 27k n=203, 450k n=346 | ESCA: 450k | READ: 27k+450k | STAD: 27k+450k)

Đặc thù của Methylation:
    - Có 2 loại chip: Illumina 27k (~27,578 probes) và 450k (~485,512 probes)
    - 27k là tập con của 450k → join='inner' tự động thu về ~27k sites chung
    - Beta value: [0, 1] — 0 = unmethylated, 1 = fully methylated
    - Một số probes không đáng tin cậy cần loại bỏ (cross-reactive & sex chromosomes)

Pipeline:
    ┌─────────────────────────────────────────────────────────────────┐
    │ BƯỚC 0 — Tải danh sách probes cần loại bỏ (file phụ trợ)        │
    │   0a. cross_reactive_probes.txt — Chen et al. 2013              │
    │       Probes bám vào nhiều vị trí → đọc sai methylation         │
    │   0b. HumanMethylation450_manifest.csv — Illumina               │
    │       Lấy danh sách probes trên chrX, chrY → bias giới tính     │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 1 — Xử lý từng file TSV (27k + 450k, 4 cancer types)       │
    │   1a. Lọc mẫu khối u nguyên phát: barcode[13:15] == '01'        │
    │   1b. Rút gọn barcode → 12 ký tự Patient ID                     │
    │   1c. Loại bỏ bệnh nhân trùng lặp (giữ Vial A = first)          │
    │   1d. Transpose → (Bệnh nhân × CpG sites)                       │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 2 — Gộp tất cả file                                        │
    │   pd.concat(join='inner') → tự thu về ~27k sites chung          │
    │   (vì 27k là tập con của 450k)                                  │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 3 — Lọc probes không đáng tin cậy                          │
    │   3a. Loại cross-reactive probes (Chen et al. 2013)             │
    │   3b. Loại probes trên chrX, chrY (Illumina manifest)           │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 4 — Quality Control                                        │
    │   Loại probe thiếu dữ liệu ở >40% bệnh nhân                     │
    │   Điền phần còn lại bằng Median Imputation                      │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 5 — Lưu kết quả                                            │
    │   → processed_methylation.csv  (Bệnh nhân × CpG sites)          │
    └─────────────────────────────────────────────────────────────────┘

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

File phụ trợ (đã có sẵn):
    - cross_reactive_probes.txt       : Chen et al. 2013, ~29,000 probes
    - HumanMethylation450_manifest.csv: Illumina 450k manifest

Lưu ý:
    - Không cố khớp số features của paper.
    - Kết quả thực tế phụ thuộc vào coverage của từng cohort GDC.
"""

import os
import glob
import argparse
import pandas as pd


# ─────────────────────────────────────────────
# 0. TẢI DANH SÁCH PROBES CẦN LỌC (FILE PHỤ TRỢ)
# ─────────────────────────────────────────────

def load_cross_reactive_probes(cross_reactive_path: str) -> set:
    """
    Bước 0a — Đọc cross-reactive probes từ file annotation Chen et al. 2013.

    File có định dạng TSV với header, các dòng bắt đầu '#' là metadata.
    Cột đầu là probe ID (IlmnID / ID), và 2 cột quan trọng:
        - AlleleA_Hits: số vị trí AlleleA bám vào in silico
        - AlleleB_Hits: số vị trí AlleleB bám vào in silico

    Probe cross-reactive khi AlleleA_Hits > 1 HOẶC AlleleB_Hits > 1
    (bám vào nhiều hơn 1 vị trí → đọc methylation sai locus)

    Args:
        cross_reactive_path: Đường dẫn file annotation Chen et al. 2013 (TSV).

    Returns:
        Set các probe ID cross-reactive cần loại bỏ.
    """
    if not cross_reactive_path or not os.path.exists(cross_reactive_path):
        print("[CpG]  ⚠ Không tìm thấy file cross-reactive probes → bỏ qua bước 3a")
        return set()

    # Đọc file: bỏ dòng comment (#), dòng đầu tiên sau comment là header
    df = pd.read_csv(cross_reactive_path, comment='#', sep='\t', low_memory=False)

    # Cột probe ID: thường là 'ID' hoặc 'IlmnID' (cột đầu tiên)
    id_col = df.columns[0]

    # Lọc cross-reactive: AlleleA_Hits > 1 HOẶC AlleleB_Hits > 1
    if 'AlleleA_Hits' in df.columns and 'AlleleB_Hits' in df.columns:
        mask = (df['AlleleA_Hits'] > 1) | (df['AlleleB_Hits'] > 1)
        probes = set(df.loc[mask, id_col].astype(str).str.strip().tolist())
        print(f"[CpG]  ✓ Bước 0a: {len(probes):,} cross-reactive probes "
              f"(AlleleA_Hits>1 hoặc AlleleB_Hits>1, từ {len(df):,} probes tổng)")
    else:
        # Fallback: nếu không có cột Hits → coi tất cả là danh sách probe cần loại
        probes = set(df.iloc[:, 0].astype(str).str.strip().tolist())
        print(f"[CpG]  ⚠ Bước 0a: Không tìm thấy cột AlleleA/B_Hits → "
              f"dùng toàn bộ {len(probes):,} probe ID (cột đầu tiên)")

    return probes


def load_sex_chromosome_probes(manifest_path: str) -> set:
    """
    Bước 0b — Đọc Illumina 450k manifest, trả về probes trên chrX và chrY.

    Probes trên chrX/Y bị ảnh hưởng bởi giới tính của bệnh nhân → tạo
    batch effect không liên quan đến methylation thực sự → cần loại bỏ.
    Manifest Illumina gốc có 7 dòng header trước khi đến header cột.

    Args:
        manifest_path: Đường dẫn file manifest CSV (Illumina gốc hoặc đã làm sạch).

    Returns:
        Set probe ID trên nhiễm sắc thể giới tính.
    """
    if not manifest_path or not os.path.exists(manifest_path):
        print("[CpG]  ⚠ Không tìm thấy Illumina manifest → bỏ qua bước 3b")
        return set()

    # File Illumina gốc có 7 dòng metadata trước header cột
    # Nếu file đã được làm sạch (bỏ metadata) thì dùng skiprows=0
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
        print("[CpG]  ⚠ Không tìm thấy cột 'CHR' trong manifest → bỏ qua bước 3b")
        return set()

    sex_probes = set(manifest[manifest['CHR'].isin(['X', 'Y'])].index.astype(str))
    print(f"[CpG]  ✓ Bước 0b: Tìm thấy {len(sex_probes):,} probes trên chrX/Y")
    return sex_probes


# ─────────────────────────────────────────────
# 1. ĐỌC VÀ XỬ LÝ TỪNG FILE TSV
# ─────────────────────────────────────────────

def clean_and_transpose(file_path: str) -> pd.DataFrame:
    """
    Đọc 1 file TSV methylation (GDC dạng matrix):
        - Rows: CpG probe ID (vd: cg00000029)
        - Cols: TCGA barcodes đầy đủ
        - Values: beta value ∈ [0, 1]

    Pipeline:
        1a. Lọc mẫu khối u nguyên phát (barcode[13:15] == '01')
        1b. Rút gọn barcode → 12 ký tự Patient ID
        1c. Loại bỏ bệnh nhân trùng lặp (giữ Vial A = first)
        1d. Transpose → (Bệnh nhân × CpG sites)

    Args:
        file_path: Đường dẫn file TSV.

    Returns:
        DataFrame shape (n_patients, n_cpg).
    """
    print(f"  -> Đọc: {os.path.basename(file_path)}")

    df = pd.read_csv(file_path, sep='\t', index_col=0, engine='c')

    # --- Bước 1a: Lọc mẫu khối u nguyên phát (-01) ---
    # TCGA barcode: TCGA-XX-XXXX-01A-... → ký tự [13:15] là sample type
    # '01' = Primary Solid Tumor, '11' = Normal tissue
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

    # --- Bước 1c: Loại bỏ bệnh nhân trùng lặp (giữ Vial A = first) ---
    df_tumor = df_tumor.loc[:, ~df_tumor.columns.duplicated(keep='first')]

    print(f"     CpG sites: {df_tumor.shape[0]:,} | Bệnh nhân: {df_tumor.shape[1]}")

    # --- Bước 1d: Transpose → (Bệnh nhân × CpG sites) ---
    return df_tumor.T


# ─────────────────────────────────────────────
# 2. LỌC PROBES KHÔNG ĐÁNG TIN CẬY
# ─────────────────────────────────────────────

def filter_unreliable_probes(
    df: pd.DataFrame,
    cross_reactive_probes: set,
    sex_chromosome_probes: set
) -> pd.DataFrame:
    """
    Bước 3 — Loại bỏ các CpG probes không đáng tin cậy trên ma trận tổng.

        3a. Cross-reactive probes (Chen et al. 2013):
            Probe bám vào nhiều vị trí trên genome → beta value đọc được
            là tổng hợp của nhiều locus → không đặc hiệu → loại bỏ.
        3b. Probes trên chrX/Y (Illumina manifest):
            Beta value phụ thuộc vào số bản sao NST giới tính của bệnh nhân
            → tạo systematic bias theo giới tính → loại bỏ.

    Args:
        df:                    DataFrame (n_patients, n_cpg).
        cross_reactive_probes: Set probe IDs cross-reactive.
        sex_chromosome_probes: Set probe IDs trên chrX/Y.

    Returns:
        DataFrame sau khi loại probes không đáng tin cậy.
    """
    n_before = df.shape[1]
    bad_probes = cross_reactive_probes | sex_chromosome_probes

    print(f"[CpG]  Bước 3 — Lọc probes không đáng tin cậy:")
    print(f"         Cross-reactive: {len(cross_reactive_probes):,} | Sex chr: {len(sex_chromosome_probes):,}")

    if bad_probes:
        keep = [col for col in df.columns if col not in bad_probes]
        df = df[keep]
        print(f"         Trước: {n_before:,} → Sau: {df.shape[1]:,} (bỏ {n_before - df.shape[1]:,} probes)")
    else:
        print("         Không có file tham chiếu → bỏ qua bước này")

    return df


# ─────────────────────────────────────────────
# 3. XỬ LÝ MISSING VALUES
# ─────────────────────────────────────────────

def handle_missing_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 4 — Quality Control trên ma trận tổng:
        4a. Loại probe thiếu dữ liệu ở >40% bệnh nhân
            (probe không đo được ở quá nhiều bệnh nhân → không đáng tin cậy)
        4b. Median Imputation cho phần còn lại
            (điền beta value trung vị của probe đó trên toàn cohort)

    Args:
        df: DataFrame (n_patients, n_cpg).

    Returns:
        DataFrame đã lọc và impute.
    """
    print(f"\n[CpG]  Kích thước trước QC: {df.shape} (Bệnh nhân × CpG sites)")

    # 4a: probe hợp lệ nếu có giá trị ở ít nhất 60% bệnh nhân
    min_valid_count = int(len(df) * 0.60)
    df_filtered = df.dropna(axis=1, thresh=min_valid_count)
    n_dropped = df.shape[1] - df_filtered.shape[1]
    if n_dropped > 0:
        print(f"[CpG]   Loại {n_dropped:,} probes thiếu >40% → còn {df_filtered.shape[1]:,} probes")

    # 4b: Điền khuyết bằng Median của từng probe
    df_imputed = df_filtered.fillna(df_filtered.median())

    print(f"[CpG]  Kích thước sau QC: {df_imputed.shape}")
    return df_imputed


# ─────────────────────────────────────────────
# 4. HÀM CHÍNH: PROCESS_CPG
# ─────────────────────────────────────────────

def process_cpg(
    input_dir: str,
    output_dir: str,
    cross_reactive_path: str = None,
    manifest_path: str = None,
) -> pd.DataFrame:
    """
    Toàn bộ pipeline xử lý DNA Methylation (GDC Harmonized, beta values).

    Args:
        input_dir:           Thư mục gốc omics (chứa subfolder 'methyl/27k/' và 'methyl/450k/')
        output_dir:          Thư mục lưu output.
        cross_reactive_path: Đường dẫn file cross-reactive probes (Chen et al. 2013).
        manifest_path:       Đường dẫn Illumina 450k manifest CSV.

    Returns:
        DataFrame (Bệnh nhân × CpG sites), đồng thời lưu ra CSV.
    """
    print("\n" + "="*60)
    print("  BẮT ĐẦU XỬ LÝ DNA METHYLATION (CpG)")
    print("  Nguồn: GDC TCGA Harmonized (27k + 450k)")
    print("="*60)

    # ── Bước 0: Tải danh sách probes cần loại bỏ ───────────────────────
    # Load sớm để thông báo ngay từ đầu, tránh chạy lâu rồi mới báo lỗi
    cross_reactive = load_cross_reactive_probes(cross_reactive_path)
    sex_chr_probes = load_sex_chromosome_probes(manifest_path)

    # ── Bước 1: Thu thập và đọc tất cả file TSV ────────────────────────
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

    df_list = []
    for fp in sorted(file_paths):
        df_cancer = clean_and_transpose(fp)
        if not df_cancer.empty:
            df_list.append(df_cancer)

    if not df_list:
        raise ValueError("Tất cả file đều rỗng sau khi xử lý. Kiểm tra lại định dạng TSV.")

    # ── Bước 2: Gộp tất cả file ─────────────────────────────────────────
    # join='inner': tự động lấy giao thoa CpG sites
    # Vì 27k là tập con của 450k, kết quả sẽ thu về ~27k sites chung
    print(f"\n[CpG]  Bước 2 — Gộp {len(df_list)} file (join='inner')...")
    master_df = pd.concat(df_list, axis=0, join='inner')
    print(f"[CpG]  Sau khi gộp: {master_df.shape} (Bệnh nhân × CpG sites, ~27k chung)")

    # ── Bước 3: Lọc probes không đáng tin cậy ──────────────────────────
    master_df = filter_unreliable_probes(master_df, cross_reactive, sex_chr_probes)

    # ── Bước 4: Quality Control ─────────────────────────────────────────
    final_df = handle_missing_and_impute(master_df)

    # ── Bước 5: Lưu output ──────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "processed_methylation.csv")
    final_df.to_csv(out_path)
    print(f"\n[CpG]  ✓ Đã lưu: {out_path}")
    print(f"[CpG]  ✓ Shape cuối: {final_df.shape}  (Bệnh nhân × CpG sites)")
    print("="*60)

    return final_df


# ─────────────────────────────────────────────
# 5. ENTRY POINT
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