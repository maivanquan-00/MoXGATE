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
        'PAM50 mRNA',  # hoặc tên cột subtype phù hợp với BRCA
    ]
    
    for file in file_paths:
        df = pd.read_csv(file, sep='\t')
        # Chỉ lấy file nếu có đủ 2 cột này
        if all(col in df.columns for col in target_cols):
            df_list.append(df[target_cols])
            
    # 3. Gộp toàn bộ lại thành Master Dataframe
    master_df = pd.concat(df_list, ignore_index=True)
    # Đổi tên cột cho ngắn gọn
    master_df.rename(columns={
        'PAM50 mRNA': 'Subtype',
    }, inplace=True)
    # Bỏ các dòng không có nhãn
    master_df = master_df.dropna(subset=['Subtype'])
    master_df = master_df.drop_duplicates(subset=['Patient ID'])
    # Chuẩn hóa tên subtype
    subtype_map = {
        'LumA': 0,
        'LumB': 1,
        'Her2': 2,
        'Basal': 3,
        'Normal': 4
    }
    master_df['Clean_Subtype'] = master_df['Subtype'].map(lambda x: str(x).replace(' ', '').replace('lumA','LumA').replace('lumB','LumB').replace('her2','Her2').replace('basal','Basal').replace('normal','Normal'))
    master_df['Target_Label'] = master_df['Clean_Subtype'].map(subtype_map)
    master_df = master_df.dropna(subset=['Target_Label'])
    master_df['Target_Label'] = master_df['Target_Label'].astype(int)
    final_labels = master_df[['Patient ID', 'Clean_Subtype', 'Target_Label']]
    return final_labels

if __name__ == "__main__":
    BASE_DIR    = "/content/drive/MyDrive/ĐATN_2025.2"
    folder_path = os.path.join(BASE_DIR, "data_original", "subtype_brca")
    output_path = os.path.join(BASE_DIR, "data_processed", "clean_labels_brca.csv")
    print("Đang xử lý dữ liệu nhãn BRCA...")
    final_labels = process_clinical_labels_brca(folder_path)
    print("Xử lý hoàn tất! 5 dòng đầu tiên:")
    print(final_labels.head())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_labels.to_csv(output_path, index=False)
    print(f"Đã lưu kết quả ra file: {output_path}")
