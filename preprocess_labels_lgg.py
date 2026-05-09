"""
preprocess_labels_lgg.py
===========================
Xử lý nhãn phân lớp cho bộ dữ liệu Brain Lower Grade Glioma (LGG).

3 phân lớp chính:
    Codel               → Label 0
    IDHmut-non-codel    → Label 1
    IDHwt               → Label 2

Dữ liệu nhãn lấy từ TCGA clinical TSV.

Cách chạy:
    !python MoXGATE/preprocess_labels_lgg.py
"""

import os
import glob
import argparse
import pandas as pd

# Mapping: Cancer Subtype → Label Index
LGG_LABEL_MAP = {
    'CODEL': 0,
    'IDHMUT-NON-CODEL': 1,
    'IDHWT': 2,
}

LGG_SUBTYPE_NAMES = {v: k for k, v in LGG_LABEL_MAP.items()}


def extract_subtype(s):
    if pd.isna(s):
        return None
    s = str(s).strip().upper()
    # Loại bỏ prefix LGG_ nếu có
    if s.startswith('LGG_'):
        s = s.replace('LGG_', '')
    
    # Normalize một số biến thể thường gặp
    if 'CODEL' in s and 'NON' not in s:
        return 'CODEL'
    if 'NON' in s and 'CODEL' in s:
        return 'IDHMUT-NON-CODEL'
    if 'IDH' in s and 'WT' in s:
        return 'IDHWT'
    return None


def process_clinical_labels_lgg(subtype_folder_path: str) -> pd.DataFrame:
    # 1. Thu thập tất cả file TSV
    tsv_files = glob.glob(os.path.join(subtype_folder_path, "*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(
            f"Không tìm thấy file .tsv nào trong: {subtype_folder_path}\n"
        )

    print(f"[LGG Labels] Tìm thấy {len(tsv_files)} file TSV.")

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
    print(f"[LGG Labels] Các cột trong file: {list(master_df.columns[:10])}")
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
    print(f"[LGG Labels] Dùng cột nhãn: '{cancer_col}'")
    
    # 4. Chuẩn hóa giá trị Subtype
    master_df['Clean_Subtype'] = master_df['Cancer_Type'].apply(extract_subtype)
    
    # 5. Lọc và dropna
    master_df = master_df.dropna(subset=['Patient_ID', 'Clean_Subtype'])
    master_df = master_df.drop_duplicates(subset=['Patient_ID'])

    # 6. Map nhãn số
    master_df['Target_Label'] = master_df['Clean_Subtype'].map(LGG_LABEL_MAP)
    master_df = master_df.dropna(subset=['Target_Label'])
    master_df['Target_Label'] = master_df['Target_Label'].astype(int)

    # 7. Chuẩn hóa Patient ID
    master_df['Patient_ID'] = master_df['Patient_ID'].str[:12]

    final_labels = master_df[['Patient_ID', 'Clean_Subtype', 'Target_Label']].copy()
    final_labels = final_labels.rename(columns={
        'Patient_ID': 'Patient ID',
        'Clean_Subtype': 'Subtype'
    })
    # Thêm cột Cancer_Type = 'LGG' (constant)
    final_labels.insert(1, 'Cancer_Type', 'LGG')

    return final_labels


if __name__ == "__main__":
    import config

    parser = argparse.ArgumentParser(
        description="Xử lý nhãn LGG từ clinical TSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--subtype_dir", type=str,
        default=config.LGG_RAW_SUBTYPE_DIR,
        help="Thư mục chứa file TSV clinical LGG",
    )
    parser.add_argument(
        "--output_path", type=str,
        default=config.LGG_LABELS_PATH,
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

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    final_labels.to_csv(args.output_path, index=False)
    print(f"\n[LGG Labels] Đã lưu: {args.output_path}")