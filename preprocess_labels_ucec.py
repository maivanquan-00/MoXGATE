"""
preprocess_labels_ucec.py
===========================
Xử lý nhãn phân lớp cho bộ dữ liệu ung thư nội mạc tử cung (UCEC).

4 phân lớp chính:
    CN_LOW  (Copy-number low)      → Label 0
    CN_HIGH (Copy-number high)     → Label 1
    MSI     (Microsatellite inst.) → Label 2
    POLE    (POLE-mutated)         → Label 3

Dữ liệu nhãn lấy từ TCGA clinical TSV.

Cách chạy:
    !python MoXGATE/preprocess_labels_ucec.py
"""

import os
import glob
import argparse
import pandas as pd

# Mapping: Cancer Subtype → Label Index
UCEC_LABEL_MAP = {
    'CN_LOW': 0,
    'CN_HIGH': 1,
    'MSI': 2,
    'POLE': 3,
}

UCEC_SUBTYPE_NAMES = {v: k for k, v in UCEC_LABEL_MAP.items()}

def extract_subtype(s):
    if pd.isna(s):
        return None
    s = str(s).strip().upper()
    # Loại bỏ prefix UCEC_ nếu có
    if s.startswith('UCEC_'):
        s = s.replace('UCEC_', '')
    
    # Normalize một số biến thể thường rặp
    if 'LOW' in s: return 'CN_LOW'
    if 'HIGH' in s: return 'CN_HIGH'
    if 'MSI' in s: return 'MSI'
    if 'POLE' in s: return 'POLE'
    return None

def process_clinical_labels_ucec(subtype_folder_path: str) -> pd.DataFrame:
    # 1. Thu thập tất cả file TSV
    tsv_files = glob.glob(os.path.join(subtype_folder_path, "*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(
            f"Không tìm thấy file .tsv nào trong: {subtype_folder_path}\n"
        )

    df_list = []
    for fpath in tsv_files:
        df = pd.read_csv(fpath, sep='\t', dtype=str)
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

    # 3. Xác định cột Subtype
    cancer_col = None
    for col in ['Subtype', 'TCGA PanCanAtlas Cancer Type Acronym', 'project_id',
                'Cancer Type Abbreviation', 'Cancer_Type', 'type']:
        if col in master_df.columns:
            cancer_col = col
            break

    if cancer_col is None:
        raise ValueError(
            f"Không tìm thấy cột Subtype trong file TSV.\n"
            f"Tất cả các cột: {list(master_df.columns)}"
        )

    master_df = master_df.rename(columns={cancer_col: 'Cancer_Type'})
    
    # 4. Chuẩn hóa giá trị Subtype
    master_df['Clean_Subtype'] = master_df['Cancer_Type'].apply(extract_subtype)
    
    # 5. Lọc và dropna
    master_df = master_df.dropna(subset=['Patient_ID', 'Clean_Subtype'])
    master_df = master_df.drop_duplicates(subset=['Patient_ID'])

    # 6. Map nhãn số
    master_df['Target_Label'] = master_df['Clean_Subtype'].map(UCEC_LABEL_MAP)
    master_df = master_df.dropna(subset=['Target_Label'])
    master_df['Target_Label'] = master_df['Target_Label'].astype(int)

    # 7. Chuẩn hóa Patient ID
    master_df['Patient_ID'] = master_df['Patient_ID'].str[:12]

    final_labels = master_df[['Patient_ID', 'Clean_Subtype', 'Target_Label']].copy()
    final_labels = final_labels.rename(columns={
        'Patient_ID': 'Patient ID',
        'Clean_Subtype': 'Subtype'
    })
    # Thêm cột Cancer_Type = 'UCEC' (constant)
    final_labels.insert(1, 'Cancer_Type', 'UCEC')

    return final_labels

if __name__ == "__main__":
    import config

    parser = argparse.ArgumentParser(
        description="Xử lý nhãn UCEC từ clinical TSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--subtype_dir", type=str,
        default=config.UCEC_RAW_SUBTYPE_DIR,
        help="Thư mục chứa file TSV clinical UCEC",
    )
    parser.add_argument(
        "--output_path", type=str,
        default=config.UCEC_LABELS_PATH,
        help="Đường dẫn file output CSV",
    )
    args = parser.parse_args()

    print("Đang xử lý dữ liệu nhãn UCEC...")
    print(f"  subtype_dir : {args.subtype_dir}")
    print(f"  output_path : {args.output_path}")

    final_labels = process_clinical_labels_ucec(args.subtype_dir)

    print("\nXử lý hoàn tất! 5 dòng đầu tiên:")
    print(final_labels.head())
    print(f"\nPhân bố subtype:\n{final_labels['Subtype'].value_counts().to_string()}")

    # Lưu ra file CSV để các file khác đọc lại
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    final_labels.to_csv(args.output_path, index=False)
    print(f"\nĐã lưu kết quả ra file: {args.output_path}")
