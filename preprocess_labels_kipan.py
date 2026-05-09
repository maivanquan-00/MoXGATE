"""
preprocess_labels_kipan.py
===========================
Xử lý nhãn phân lớp cho bộ dữ liệu ung thư thận (KIPAN).

3 phân lớp chính:
    KICH  (Kidney Chromophobe)             — 65  mẫu → Label 0
    KIRC  (Kidney Renal Clear Cell)        — 352 mẫu → Label 1
    KIRP  (Kidney Renal Papillary)         — 271 mẫu → Label 2

Dữ liệu nhãn lấy từ TCGA clinical TSV.
File TSV cần có cột 'Patient ID' và 'TCGA PanCanAtlas Cancer Type Acronym'
(hoặc cột 'Subtype' tùy nguồn tải về).

Cách chạy:
    !python MoXGATE/preprocess_labels_kipan.py
    !python MoXGATE/preprocess_labels_kipan.py \\
        --subtype_dir /path/to/subtype_kipan \\
        --output_path /path/to/clean_labels_kipan.csv
"""

import os
import glob
import argparse
import pandas as pd


# Mapping: Cancer Type Acronym → Label Index
KIPAN_LABEL_MAP = {
    'KICH': 0,
    'KIRC': 1,
    'KIRP': 2,
}

KIPAN_SUBTYPE_NAMES = {v: k for k, v in KIPAN_LABEL_MAP.items()}


def process_clinical_labels_kipan(subtype_folder_path: str) -> pd.DataFrame:
    """
    Đọc và xử lý clinical data cho KIPAN (Kidney Pan-Cancer).

    Chiến lược:
        - Tìm tất cả file .tsv trong subtype_folder_path.
        - Ghép thành master_df.
        - Lấy cột 'TCGA PanCanAtlas Cancer Type Acronym' làm nhãn.
        - Map KICH→0, KIRC→1, KIRP→2.
        - Lọc bỏ các loại ung thư khác nếu có trong file.

    Returns:
        DataFrame với cột: Patient ID, Cancer_Type, Target_Label
    """
    # 1. Thu thập tất cả file TSV
    tsv_files = glob.glob(os.path.join(subtype_folder_path, "*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(
            f"Không tìm thấy file .tsv nào trong: {subtype_folder_path}\n"
            f"Hãy chắc chắn thư mục chứa clinical data từ TCGA/GDC."
        )

    print(f"[KIPAN Labels] Tìm thấy {len(tsv_files)} file TSV.")

    df_list = []
    for fpath in tsv_files:
        df = pd.read_csv(fpath, sep='\t', dtype=str)
        # Chuẩn hóa tên cột về lowercase để dễ xử lý
        df.columns = df.columns.str.strip()
        df_list.append(df)

    master_df = pd.concat(df_list, ignore_index=True)

    # 2. Xác định cột Patient ID
    # Thứ tự ưu tiên: 'Patient ID' > 'case_id' > cột đầu tiên
    if 'Patient ID' in master_df.columns:
        master_df = master_df.rename(columns={'Patient ID': 'Patient_ID'})
    elif 'case_id' in master_df.columns:
        master_df = master_df.rename(columns={'case_id': 'Patient_ID'})
    else:
        master_df = master_df.rename(columns={master_df.columns[0]: 'Patient_ID'})

    # 3. Xác định cột Cancer Type
    # Thứ tự ưu tiên theo định dạng TCGA phổ biến:
    #   'Subtype'                              → 3 file KICH/KIRC/KIRP riêng biệt
    #   'TCGA PanCanAtlas Cancer Type Acronym' → Pan-Cancer clinical TSV
    #   'project_id'                           → GDC manifest (dạng TCGA-KIRC)
    #   'Cancer Type Abbreviation'             → các nguồn khác
    print(f"[KIPAN Labels] Các cột trong file: {list(master_df.columns[:10])}")
    cancer_col = None
    for col in ['Subtype', 'TCGA PanCanAtlas Cancer Type Acronym', 'project_id',
                'Cancer Type Abbreviation', 'Cancer_Type', 'type']:
        if col in master_df.columns:
            cancer_col = col
            break

    if cancer_col is None:
        raise ValueError(
            f"Không tìm thấy cột Cancer Type trong file TSV.\n"
            f"Tất cả các cột: {list(master_df.columns)}"
        )

    master_df = master_df.rename(columns={cancer_col: 'Cancer_Type'})
    print(f"[KIPAN Labels] Dùng cột nhãn: '{cancer_col}'")
    print(f"[KIPAN Labels] Mẫu giá trị Cancer_Type: {master_df['Cancer_Type'].unique()[:10].tolist()}")

    # 4. Chuẩn hóa giá trị Cancer_Type
    # project_id thường có dạng 'TCGA-KIRC' → tách phần sau dấu '-'
    master_df['Cancer_Type'] = master_df['Cancer_Type'].astype(str).str.strip()
    master_df['Cancer_Type'] = master_df['Cancer_Type'].apply(
        lambda x: x.split('-')[-1] if '-' in x else x
    ).str.upper()

    # 5. Chỉ giữ 3 loại KICH, KIRC, KIRP
    master_df = master_df[master_df['Cancer_Type'].isin(KIPAN_LABEL_MAP.keys())]
    master_df = master_df.dropna(subset=['Patient_ID'])
    master_df = master_df.drop_duplicates(subset=['Patient_ID'])

    # 6. Map nhãn số
    master_df['Target_Label'] = master_df['Cancer_Type'].map(KIPAN_LABEL_MAP)
    master_df = master_df.dropna(subset=['Target_Label'])
    master_df['Target_Label'] = master_df['Target_Label'].astype(int)

    # 7. Chuẩn hóa Patient ID về định dạng TCGA-XX-XXXX
    # GDC thường dùng 12 ký tự đầu: TCGA-A3-3308
    master_df['Patient_ID'] = master_df['Patient_ID'].str[:12]

    # Thêm cột Subtype = Cancer_Type (để đồng bộ với các dataset khác)
    master_df['Subtype'] = master_df['Cancer_Type']
    
    final_labels = master_df[['Patient_ID', 'Cancer_Type', 'Subtype', 'Target_Label']].copy()
    final_labels = final_labels.rename(columns={'Patient_ID': 'Patient ID'})

    return final_labels


if __name__ == "__main__":
    import config

    parser = argparse.ArgumentParser(
        description="Xử lý nhãn KIPAN từ clinical TSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--subtype_dir", type=str,
        default=config.KIPAN_RAW_SUBTYPE_DIR,
        help="Thư mục chứa file TSV clinical KIPAN",
    )
    parser.add_argument(
        "--output_path", type=str,
        default=config.KIPAN_LABELS_PATH,
        help="Đường dẫn file output CSV",
    )
    args = parser.parse_args()

    print("Đang xử lý dữ liệu nhãn KIPAN...")
    print(f"  subtype_dir : {args.subtype_dir}")
    print(f"  output_path : {args.output_path}")

    final_labels = process_clinical_labels_kipan(args.subtype_dir)

    print("\nXử lý hoàn tất! 5 dòng đầu tiên:")
    print(final_labels.head())
    print(f"\nPhân bố subtype:\n{final_labels['Subtype'].value_counts().to_string()}")

    # Lưu ra file CSV để các file khác đọc lại
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    final_labels.to_csv(args.output_path, index=False)
    print(f"\nĐã lưu kết quả ra file: {args.output_path}")
