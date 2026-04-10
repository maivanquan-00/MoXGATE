import pandas as pd
import os
import glob

def process_clinical_labels_brca(subtype_folder_path):
    """
    Đọc và xử lý clinical data cho BRCA (Breast Cancer) — PAM50 subtypes.
    """
    # 1. Tìm tất cả các file .tsv
    file_paths = glob.glob(os.path.join(subtype_folder_path, "*.tsv"))
    df_list = []
    
    # 2. Các cột mục tiêu cần lấy từ file gốc
    target_cols = [
        'Patient ID',
        'Subtype',
    ]
    for file in file_paths:
        df = pd.read_csv(file, sep='\t')
        if all(col in df.columns for col in target_cols):
            df_list.append(df[target_cols])

    master_df = pd.concat(df_list, ignore_index=True)
    master_df = master_df.dropna(subset=['Subtype'])
    master_df = master_df.drop_duplicates(subset=['Patient ID'])

    # Tách hậu tố sau dấu _ (BRCA_LumA -> LumA)
    def extract_subtype(s):
        if pd.isna(s):
            return None
        parts = str(s).split('_')
        return parts[-1] if len(parts) > 1 else parts[0]

    master_df['Clean_Subtype'] = master_df['Subtype'].map(extract_subtype)

    # Chuẩn hóa tên subtype
    subtype_map = {
        'LumA': 0,
        'LumB': 1,
        'Her2': 2,
        'Basal': 3,
        'Normal': 4
    }
    master_df['Clean_Subtype'] = master_df['Clean_Subtype'].map(lambda x: str(x).replace(' ', '').replace('lumA','LumA').replace('lumB','LumB').replace('her2','Her2').replace('basal','Basal').replace('normal','Normal'))
    master_df['Target_Label'] = master_df['Clean_Subtype'].map(subtype_map)
    master_df = master_df.dropna(subset=['Target_Label'])
    master_df['Target_Label'] = master_df['Target_Label'].astype(int)
    final_labels = master_df[['Patient ID', 'Clean_Subtype', 'Target_Label']]
    return final_labels

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Xử lý nhãn BRCA từ clinical TSV")
    parser.add_argument(
        "--subtype_dir", type=str,
        default=r"d:\ĐATN\MoXGATE\data_original\subtype_brca",
        help="Thư mục chứa file TSV clinical BRCA (mặc định: data_original/subtype_brca/)",
    )
    parser.add_argument(
        "--output_path", type=str,
        default=r"d:\ĐATN\MoXGATE\data_processed_brca\clean_labels_brca.csv",
        help="Đường dẫn file output CSV (mặc định: data_processed_brca/clean_labels_brca.csv)",
    )
    args = parser.parse_args()

    print("Đang xử lý dữ liệu nhãn BRCA...")
    print(f"  subtype_dir : {args.subtype_dir}")
    print(f"  output_path : {args.output_path}")

    final_labels = process_clinical_labels_brca(args.subtype_dir)

    print("\nXử lý hoàn tất! 5 dòng đầu tiên:")
    print(final_labels.head())
    print(f"\nPhân bố subtype:\n{final_labels['Clean_Subtype'].value_counts().to_string()}")

    # Lưu ra file CSV để các file khác đọc lại
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    final_labels.to_csv(args.output_path, index=False)
    print(f"\nĐã lưu kết quả ra file: {args.output_path}")
