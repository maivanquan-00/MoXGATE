import pandas as pd
import os
import glob

def process_clinical_labels(subtype_folder_path):
    """
    Hàm đọc và xử lý toàn bộ file clinical data từ thư mục subtype.
    """
    # 1. Tìm tất cả các file .tsv
    file_paths = glob.glob(os.path.join(subtype_folder_path, "*.tsv"))
    df_list = []
    
    # 2. Các cột mục tiêu cần lấy từ file gốc
    target_cols = [
        'Patient ID', 
        'TCGA PanCanAtlas Cancer Type Acronym', 
        'Subtype'
    ]
    
    for file in file_paths:
        df = pd.read_csv(file, sep='\t')
        
        # Chỉ lấy file nếu có đủ 3 cột này (tránh lỗi file hỏng/sai format)
        if all(col in df.columns for col in target_cols):
            df_list.append(df[target_cols])
            
    # 3. Gộp toàn bộ lại thành Master Dataframe
    master_df = pd.concat(df_list, ignore_index=True)
    
    # Đổi tên cột cho ngắn gọn, dễ code
    master_df.rename(columns={
        'TCGA PanCanAtlas Cancer Type Acronym': 'Cancer_Type'
    }, inplace=True)
    
    # 4. Làm sạch dữ liệu
    # Bỏ các dòng không có nhãn (Subtype bị NaN)
    master_df = master_df.dropna(subset=['Subtype'])
    # Bỏ trùng lặp bệnh nhân
    master_df = master_df.drop_duplicates(subset=['Patient ID'])
    
    # 5. XỬ LÝ NHÃN (LABEL ENGINEERING)
    # 5a. Gọt bỏ tiền tố loại ung thư TRƯỚC (Ví dụ: "ESCA_CIN" -> "CIN", "COAD_POLE" -> "POLE")
    def clean_prefix(row):
        subtype_val = str(row['Subtype'])
        prefix = str(row['Cancer_Type']) + '_' # Tạo chuỗi "ESCA_" hoặc "COAD_" ...
        if subtype_val.startswith(prefix):
            return subtype_val.replace(prefix, '')
        return subtype_val
        
    master_df['Clean_Subtype'] = master_df.apply(clean_prefix, axis=1)
    
    # 5b. Chuẩn hóa mọi biến thể đặc thù về chuẩn chung
    # Lúc này, dù là ESCA_POLE hay COAD_POLE thì cũng đã bị gọt thành chữ 'POLE'
    # Ta chỉ cần 1 lệnh duy nhất để quy tất cả POLE về HM-SNV
    master_df['Clean_Subtype'] = master_df['Clean_Subtype'].replace('POLE', 'HM-SNV')
    
    # 5c. Mã hóa nhãn thành số nguyên (One-hot / Label Encoding cho Deep Learning)
    label_dict = {
        'CIN': 0,
        'GS': 1,
        'MSI': 2,
        'HM-SNV': 3,
        'EBV': 4
    }
    master_df['Target_Label'] = master_df['Clean_Subtype'].map(label_dict)
    
    # Lọc bỏ những bệnh nhân thuộc các Subtype rác (nếu có)
    master_df = master_df.dropna(subset=['Target_Label'])
    master_df['Target_Label'] = master_df['Target_Label'].astype(int)
    
    # Trả về bảng cuối cùng chỉ với các cột cần thiết
    final_labels = master_df[['Patient ID', 'Cancer_Type', 'Clean_Subtype', 'Target_Label']]

    return final_labels