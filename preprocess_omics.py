import pandas as pd
import os
import glob

import re

def get_protein_coding_list(gtf_path):
    """
    Trích xuất danh sách Gene Symbols mã hóa protein từ file GENCODE GTF.
    Sử dụng streaming để tiết kiệm RAM.
    """
    print(f" -> Đang quét file GTF để tìm gen mã hóa protein: {os.path.basename(gtf_path)}")
    protein_coding_genes = set()
    
    # Sử dụng Regex để bắt gene_name và gene_type nhanh hơn
    # GENCODE v36 format: gene_type "protein_coding"; ... gene_name "TP53";
    with open(gtf_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue # Bỏ qua dòng chú thích
            
            # Chỉ xét các dòng định nghĩa 'gene' để tăng tốc
            parts = line.split('\t')
            if len(parts) > 2 and parts[2] == 'gene':
                attributes = parts[8]
                if 'gene_type "protein_coding"' in attributes:
                    # Tìm gene_name trong chuỗi attributes
                    match = re.search(r'gene_name "([^"]+)"', attributes)
                    if match:
                        protein_coding_genes.add(match.group(1))
                        
    print(f" -> Đã tìm thấy {len(protein_coding_genes)} gen mã hóa protein trong GENCODE.")
    return list(protein_coding_genes)

def clean_and_transpose(file_path):
    """
    Đọc 1 file TSV Omics: Lọc u nguyên phát (-01), gọt ID và chuyển vị ma trận.
    """
    print(f"    -> Đọc: {os.path.basename(file_path)}")
    # engine='c' giúp tăng tốc độ đọc các file TSV cực lớn
    df = pd.read_csv(file_path, sep='\t', index_col=0, engine='c')
    
    # 1. Bắt đúng nhịp Barcode: Lọc khối u nguyên phát (-01)
    tumor_cols = [col for col in df.columns if isinstance(col, str) and len(col) >= 15 and col[13:15] == '01']
    df_tumor = df[tumor_cols]
    
    # 2. Gọt Barcode về 12 ký tự (Patient ID)
    df_tumor.columns = [col[:12] for col in df_tumor.columns]
    
    # 3. Lọc trùng lặp (giữ lọ đầu tiên - Vial A)
    df_tumor = df_tumor.loc[:, ~df_tumor.columns.duplicated(keep='first')]
    
    # 4. Transpose: Chuyển Hàng thành Bệnh nhân, Cột thành Đặc trưng
    return df_tumor.T

def handle_missing_and_impute(df):
    """
    Loại bỏ các đặc trưng (cột) thiếu > 40% dữ liệu.
    Phần còn lại điền khuyết bằng trung vị (Median Imputation).
    """
    print(f"    -> Kích thước trước lọc: {df.shape} (Bệnh nhân x Đặc trưng)")
    
    # Số lượng dữ liệu hợp lệ tối thiểu = 60% tổng số bệnh nhân
    min_valid_count = int(len(df) * 0.60)
    
    # Bỏ các cột (axis=1) không đạt chuẩn
    df_filtered = df.dropna(axis=1, thresh=min_valid_count)
    
    # Điền khuyết bằng Median
    df_imputed = df_filtered.fillna(df_filtered.median())
    
    print(f"    -> Kích thước sau lọc và điền khuyết: {df_imputed.shape}")
    return df_imputed

def process_standard_omics(folder_path, omics_name):
    """
    Luồng xử lý chung cho Gene và miRNA
    """
    print(f"\n[BẮT ĐẦU XỬ LÝ {omics_name.upper()}]")
    file_paths = glob.glob(os.path.join(folder_path, "*.tsv"))
    df_list = []
    
    for file in file_paths:
        df_list.append(clean_and_transpose(file))
        
    print(f"  => Đang gộp {len(df_list)} file và lấy giao thoa đặc trưng...")
    # join='inner' ép Pandas chỉ giữ lại các Cột (Gen/miRNA) chung nhất giữa 4 loại ung thư
    master_df = pd.concat(df_list, axis=0, join='inner')
    
    # Xử lý missing values trên ma trận tổng
    final_df = handle_missing_and_impute(master_df)
    return final_df

def process_methylation(methyl_folder_path):
    """
    Luồng xử lý đặc thù cho Methylation (Giao thoa chip 27k và 450k)
    """
    print(f"\n[BẮT ĐẦU XỬ LÝ METHYLATION]")
    path_27k = os.path.join(methyl_folder_path, "27k", "*.tsv")
    path_450k = os.path.join(methyl_folder_path, "450k", "*.tsv")
    
    # Gom tất cả đường dẫn file của cả 2 loại chip
    file_paths = glob.glob(path_27k) + glob.glob(path_450k)
    df_list = []
    
    for file in file_paths:
        df_list.append(clean_and_transpose(file))
        
    print(f"  => Đang gộp {len(file_paths)} file (giao thoa chip 27k & 450k)...")
    # Phép thuật nằm ở đây: join='inner' sẽ ép các file 450k khổng lồ
    # thu nhỏ lại đúng bằng với số lượng CpG của các file 27k.
    master_df = pd.concat(df_list, axis=0, join='inner')
    
    final_df = handle_missing_and_impute(master_df)
    return final_df

if __name__ == "__main__":
    # KHAI BÁO ĐƯỜNG DẪN THƯ MỤC
    BASE_DIR = r"D:\ĐATN\MoXGATE\data_original\multi_omics"
    OUT_DIR = r"D:\ĐATN\MoXGATE\data_processed"
    
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Xử lý Gene
    gene_path = os.path.join(BASE_DIR, "gene")
    df_gene = process_standard_omics(gene_path, "Gene")
    df_gene.to_csv(os.path.join(OUT_DIR, "processed_gene.csv"))
    
    # 2. Xử lý miRNA
    mirna_path = os.path.join(BASE_DIR, "mirna")
    df_mirna = process_standard_omics(mirna_path, "miRNA")
    df_mirna.to_csv(os.path.join(OUT_DIR, "processed_mirna.csv"))
    
    # 3. Xử lý Methylation
    methyl_path = os.path.join(BASE_DIR, "methyl")
    df_methyl = process_methylation(methyl_path)
    df_methyl.to_csv(os.path.join(OUT_DIR, "processed_methylation.csv"))
    
    print("\n[HOÀN TẤT!] Toàn bộ ma trận Omics đã được lưu vào thư mục data_processed.")