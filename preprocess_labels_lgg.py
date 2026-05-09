"""
preprocess_labels_lgg.py
===========================
Xử lý nhãn phân lớp cho bộ dữ liệu Brain Lower Grade Glioma (LGG).

3 phân lớp chính (theo đề xuất):
    Codel               → Label 0
    IDHmut-non-codel    → Label 1
    IDHwt               → Label 2

Dữ liệu nhãn lấy từ TCGA clinical TSV chứa cột định danh subtype.
File TSV cần có cột 'Patient ID' và 'Subtype' (hoặc các biến thể tương đương).

Cách chạy:
    !python MoXGATE/preprocess_labels_lgg.py
    !python MoXGATE/preprocess_labels_lgg.py \\
        --subtype_dir /path/to/subtype_lgg \\
        --output_path /path/to/clean_labels_lgg.csv
"""

import os
import glob
import argparse
import pandas as pd


# Mapping: Subtype → Label Index
LGG_LABEL_MAP = {
    'CODEL': 0,
    'IDHMUT-NON-CODEL': 1,
    'IDHWT': 2,
}

LGG_SUBTYPE_NAMES = {v: k for k, v in LGG_LABEL_MAP.items()}


def process_clinical_labels_lgg(subtype_folder_path: str) -> pd.DataFrame:
    """
    Đọc và xử lý clinical data cho LGG.

    Chiến lược:
        - Tìm tất cả file .tsv trong subtype_folder_path.
        - Ghép thành master_df.
        - Lấy cột phân loại (Subtype) làm nhãn.
        - Map thành các nhãn tương ứng 0, 1, 2.

    Returns:
        DataFrame với cột: Patient ID, Cancer_Type, Target_Label
    """
    if not os.path.exists(subtype_folder_path):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {subtype_folder_path}")

    # 1. Thu thập tất cả file TSV
    tsv_files = glob.glob(os.path.join(subtype_folder_path, "*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(
            f"Không tìm thấy file .tsv nào trong: {subtype_folder_path}\n"
            f"Hãy chắc chắn thư mục chứa clinical data từ TCGA/GDC."
        )

    print(f"[LGG Labels] Tìm thấy {len(tsv_files)} file TSV.")

    df_list = []
    for fpath in tsv_files:
        df = pd.read_csv(fpath, sep='\t', dtype=str)
        # Chuẩn hóa tên cột về lowercase để dễ xử lý
        df.columns = df.columns.str.strip()
        df_list.append(df)

    master_df = pd.concat(df_list, ignore_index=True)

    # 2. Xác định cột Patient ID
    if 'Patient ID' in master_df.columns:
        master_df = master_df.rename(columns={'Patient ID': 'Patient_ID'})
    elif 'case_id' in master_df.columns:
        master_df = master_df.rename(columns={'case_id': 'Patient_ID'})
    else:
        master_df = master_df.rename(columns={master_df.columns[0]: 'Patient_ID'})

    # 3. Xác định cột chứa Subtype
    print(f"[LGG Labels] Các cột trong file: {list(master_df.columns[:10])}")
    cancer_col = None
    # Tìm kiếm các cột mang ý nghĩa subtype cho LGG
    for col in ['Subtype', 'Transcriptome Subtype', 'Molecular subtype', 'project_id',
                'Cancer Type Abbreviation', 'Cancer_Type', 'type']:
        if col in master_df.columns:
            cancer_col = col
            break

    if cancer_col is None:
        raise ValueError(
            f"Không tìm thấy cột Cancer Type/Subtype trong file TSV.\n"
            f"Tất cả các cột: {list(master_df.columns)}"
        )

    master_df = master_df.rename(columns={cancer_col: 'Cancer_Type'})
    print(f"[LGG Labels] Dùng cột nhãn: '{cancer_col}'")
    print(f"[LGG Labels] Mẫu giá trị ban đầu: {master_df['Cancer_Type'].unique()[:10].tolist()}")

    # 4. Chuẩn hóa giá trị Cancer_Type
    master_df['Cancer_Type'] = master_df['Cancer_Type'].astype(str).str.strip().str.upper()

    # Thử mapping một số phiên bản phổ biến để khớp với key CODEL, IDHMUT-NON-CODEL, IDHWT
    def normalize_lgg_subtype(val):
        """Map các dạng chữ khác nhau về chuẩn chung"""
        v = val.upper().replace(' ', '').replace('_', '-')
        if 'CODEL' in v and 'NON' not in v:
            return 'CODEL'
        elif 'IDHMUT' in v and ('NONCODEL' in v or 'NON-CODEL' in v):
            return 'IDHMUT-NON-CODEL'
        elif 'IDHWT' in v:
            return 'IDHWT'
        return val # Giữ nguyên nếu không khớp

    master_df['Cancer_Type'] = master_df['Cancer_Type'].apply(normalize_lgg_subtype)

    # 5. Chỉ giữ các class hợp lệ
    master_df = master_df[master_df['Cancer_Type'].isin(LGG_LABEL_MAP.keys())]
    master_df = master_df.dropna(subset=['Patient_ID'])
    master_df = master_df.drop_duplicates(subset=['Patient_ID'])

    # 6. Map nhãn số
    master_df['Target_Label'] = master_df['Cancer_Type'].map(LGG_LABEL_MAP)
    master_df = master_df.dropna(subset=['Target_Label'])
    master_df['Target_Label'] = master_df['Target_Label'].astype(int)

    # 7. Chuẩn hóa Patient ID về định dạng TCGA-XX-XXXX
    master_df['Patient_ID'] = master_df['Patient_ID'].str[:12]

    final_labels = master_df[['Patient_ID', 'Cancer_Type', 'Target_Label']].copy()
    final_labels = final_labels.rename(columns={'Patient_ID': 'Patient ID'})

    return final_labels


if __name__ == "__main__":
    # Thử lấy config tự định nghĩa, tránh lỗi nếu file config không có biến này
    try:
        import config
        default_subtype_dir = getattr(config, 'LGG_RAW_SUBTYPE_DIR', 'data_original/subtype_lgg')
        default_out_path = getattr(config, 'LGG_LABELS_PATH', 'clean_labels_lgg.csv')
    except ImportError:
        default_subtype_dir = 'data_original/subtype_lgg'
        default_out_path = 'clean_labels_lgg.csv'

    parser = argparse.ArgumentParser(
        description="Xử lý nhãn LGG từ clinical TSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--subtype_dir", type=str,
        default=default_subtype_dir,
        help="Thư mục chứa file TSV clinical LGG",
    )
    parser.add_argument(
        "--output_path", type=str,
        default=default_out_path,
        help="Đường dẫn file output CSV",
    )
    args = parser.parse_args()

    print(f"[LGG Labels] Đọc từ  : {args.subtype_dir}")
    print(f"[LGG Labels] Lưu ra  : {args.output_path}")

    final_labels = process_clinical_labels_lgg(args.subtype_dir)

    print(f"\n[LGG Labels] Tổng số bệnh nhân: {len(final_labels)}")
    print(f"\n[LGG Labels] Phân bố subtype:")
    dist = final_labels.groupby(['Cancer_Type', 'Target_Label']).size().reset_index(name='count')
    for _, row in dist.iterrows():
        print(f"  {row['Cancer_Type']} (label={row['Target_Label']}): {row['count']} mẫu")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    final_labels.to_csv(args.output_path, index=False)
    print(f"\n[LGG Labels] Đã lưu: {args.output_path}")