# import pandas as pd
# import os
# import glob
# import re
# import argparse

# def get_protein_coding_list(gtf_path):
#     """
#     Trích xuất danh sách Gene Symbols mã hóa protein từ file GENCODE GTF.
#     Sử dụng streaming để tiết kiệm RAM.
#     """
#     print(f" -> Đang quét file GTF để tìm gen mã hóa protein: {os.path.basename(gtf_path)}")
#     protein_coding_genes = set()
    
#     # Sử dụng Regex để bắt gene_name và gene_type nhanh hơn
#     # GENCODE v36 format: gene_type "protein_coding"; ... gene_name "TP53";
#     with open(gtf_path, 'r') as f:
#         for line in f:
#             if line.startswith('#'): continue # Bỏ qua dòng chú thích
            
#             # Chỉ xét các dòng định nghĩa 'gene' để tăng tốc
#             parts = line.split('\t')
#             if len(parts) > 2 and parts[2] == 'gene':
#                 attributes = parts[8]
#                 if 'gene_type "protein_coding"' in attributes:
#                     # Tìm gene_name trong chuỗi attributes
#                     # Đổi từ gene_name sang gene_id
#                     match = re.search(r'gene_id "([^"]+)"', attributes)
#                     if match:
#                         raw_id = match.group(1) # Lấy được "ENSG00000141510.16"
#                         # Gọt bỏ phần đuôi thập phân (phiên bản), chỉ giữ lại phần gốc
#                         clean_id = raw_id.split('.')[0] 
#                         protein_coding_genes.add(clean_id)
                        
#     print(f" -> Đã tìm thấy {len(protein_coding_genes)} gen mã hóa protein trong GENCODE.")
#     return list(protein_coding_genes)

# def clean_and_transpose(file_path):
#     """
#     Đọc 1 file TSV Omics: Lọc u nguyên phát (-01), gọt ID và chuyển vị ma trận.
#     """
#     print(f"    -> Đọc: {os.path.basename(file_path)}")
#     # engine='c' giúp tăng tốc độ đọc các file TSV cực lớn
#     df = pd.read_csv(file_path, sep='\t', index_col=0, engine='c')
    
#     # 1. Bắt đúng nhịp Barcode: Lọc khối u nguyên phát (-01)
#     tumor_cols = [col for col in df.columns if isinstance(col, str) and len(col) >= 15 and col[13:15] == '01']
#     df_tumor = df[tumor_cols]
    
#     # 2. Gọt Barcode về 12 ký tự (Patient ID)
#     df_tumor.columns = [col[:12] for col in df_tumor.columns]
    
#     # 3. Lọc trùng lặp (giữ lọ đầu tiên - Vial A)
#     df_tumor = df_tumor.loc[:, ~df_tumor.columns.duplicated(keep='first')]
    
#     # 4. Transpose: Chuyển Hàng thành Bệnh nhân, Cột thành Đặc trưng
#     return df_tumor.T

# def handle_missing_and_impute(df):
#     """
#     Loại bỏ các đặc trưng (cột) thiếu > 40% dữ liệu.
#     Phần còn lại điền khuyết bằng trung vị (Median Imputation).
#     """
#     print(f"    -> Kích thước trước lọc: {df.shape} (Bệnh nhân x Đặc trưng)")
    
#     # Số lượng dữ liệu hợp lệ tối thiểu = 60% tổng số bệnh nhân
#     min_valid_count = int(len(df) * 0.60)
    
#     # Bỏ các cột (axis=1) không đạt chuẩn
#     df_filtered = df.dropna(axis=1, thresh=min_valid_count)
    
#     # Điền khuyết bằng Median
#     df_imputed = df_filtered.fillna(df_filtered.median())
    
#     print(f"    -> Kích thước sau lọc và điền khuyết: {df_imputed.shape}")
#     return df_imputed

# def process_standard_omics(folder_path, omics_name, protein_coding_list=None):
#     print(f"\n[BẮT ĐẦU XỬ LÝ {omics_name.upper()}]")
#     file_paths = glob.glob(os.path.join(folder_path, "*.tsv"))
#     df_list = []
    
#     for file in file_paths:
#         df_list.append(clean_and_transpose(file))
        
#     print(f"  => Đang gộp {len(df_list)} file và lấy giao thoa đặc trưng...")
#     master_df = pd.concat(df_list, axis=0, join='inner')
    
# # --- MỚI: BỘ LỌC PROTEIN CODING CHO GENE ---
#     if omics_name.lower() == 'gene' and protein_coding_list is not None:
#         print(f"  => Đang chuẩn hóa định dạng ID gen (gọt đuôi version)...")
#         # Chuyển đổi tên cột: 'ENSG00000141510.11' -> 'ENSG00000141510'
#         master_df.columns = [str(col).split('.')[0] for col in master_df.columns]
        
#         print(f"  => Đang lấy giao thoa với Protein-coding genes...")
#         valid_genes = master_df.columns.intersection(protein_coding_list)
#         master_df = master_df[valid_genes]
#         print(f"  => Còn lại {len(valid_genes)} đặc trưng sau lọc sinh học.")
#     # -------------------------------------------

#     final_df = handle_missing_and_impute(master_df)
#     return final_df

# def process_methylation(methyl_folder_path):
#     """
#     Luồng xử lý đặc thù cho Methylation (Giao thoa chip 27k và 450k)
#     """
#     print(f"\n[BẮT ĐẦU XỬ LÝ METHYLATION]")
#     path_27k = os.path.join(methyl_folder_path, "27k", "*.tsv")
#     path_450k = os.path.join(methyl_folder_path, "450k", "*.tsv")
    
#     # Gom tất cả đường dẫn file của cả 2 loại chip
#     file_paths = glob.glob(path_27k) + glob.glob(path_450k)
#     df_list = []
    
#     for file in file_paths:
#         df_list.append(clean_and_transpose(file))
        
#     print(f"  => Đang gộp {len(file_paths)} file (giao thoa chip 27k & 450k)...")
#     # Phép thuật nằm ở đây: join='inner' sẽ ép các file 450k khổng lồ
#     # thu nhỏ lại đúng bằng với số lượng CpG của các file 27k.
#     master_df = pd.concat(df_list, axis=0, join='inner')
    
#     final_df = handle_missing_and_impute(master_df)
#     return final_df

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_dir', type=str, required=True)
#     parser.add_argument('--output_dir', type=str, required=True)
#     # Thêm tham số cho file GTF
#     parser.add_argument('--gtf_path', type=str, default=None) 
#     args = parser.parse_args()
    
#     BASE_DIR = args.input_dir
#     OUT_DIR = args.output_dir
    
#     # 0. Đọc danh sách Protein Coding trước (nếu có file GTF)
#     pc_list = None
#     if args.gtf_path:
#         pc_list = get_protein_coding_list(args.gtf_path)
    
#     # 1. Xử lý Gene (Truyền pc_list vào)
#     gene_path = os.path.join(BASE_DIR, "gene")
#     df_gene = process_standard_omics(gene_path, "Gene", protein_coding_list=pc_list)
#     df_gene.to_csv(os.path.join(OUT_DIR, "processed_gene.csv"))
    
#     # 2. Xử lý miRNA
#     mirna_path = os.path.join(BASE_DIR, "mirna")
#     df_mirna = process_standard_omics(mirna_path, "miRNA")
#     df_mirna.to_csv(os.path.join(OUT_DIR, "processed_mirna.csv"))
    
#     # 3. Xử lý Methylation
#     methyl_path = os.path.join(BASE_DIR, "methyl")
#     df_methyl = process_methylation(methyl_path)
#     df_methyl.to_csv(os.path.join(OUT_DIR, "processed_methylation.csv"))
    
#     print("\n[HOÀN TẤT!] Toàn bộ ma trận Omics đã được lưu vào thư mục data_processed.")


"""
Copilot
"""

import pandas as pd
import os
import glob
import re
import argparse

def get_protein_coding_list(gtf_path):
    """
    Trích xuất danh sách gene_id mã hóa protein từ file GENCODE GTF.
    """
    print(f" -> Đang quét file GTF: {os.path.basename(gtf_path)}")
    protein_coding_genes = set()
    with open(gtf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) > 2 and parts[2] == 'gene':
                attributes = parts[8]
                if 'gene_type "protein_coding"' in attributes:
                    match = re.search(r'gene_id "([^"]+)"', attributes)
                    if match:
                        raw_id = match.group(1)
                        clean_id = raw_id.split('.')[0]
                        protein_coding_genes.add(clean_id)
    print(f" -> Tìm thấy {len(protein_coding_genes)} gene protein-coding.")
    return list(protein_coding_genes)

def clean_and_transpose(file_path):
    """
    Đọc 1 file TSV Omics: lọc khối u nguyên phát (-01), gọt ID và chuyển vị ma trận.
    """
    print(f"    -> Đọc: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path, sep='\t', index_col=0, engine='c')
    tumor_cols = [col for col in df.columns if isinstance(col, str) and len(col) >= 15 and col[13:15] == '01']
    df_tumor = df[tumor_cols]
    df_tumor.columns = [col[:12] for col in df_tumor.columns]
    df_tumor = df_tumor.loc[:, ~df_tumor.columns.duplicated(keep='first')]
    return df_tumor.T

def handle_missing_and_impute(df):
    """
    Loại bỏ đặc trưng thiếu >40% dữ liệu, sau đó điền khuyết bằng median.
    """
    print(f"    -> Kích thước trước lọc: {df.shape}")
    min_valid_count = int(len(df) * 0.60)
    df_filtered = df.dropna(axis=1, thresh=min_valid_count)
    df_imputed = df_filtered.fillna(df_filtered.median())
    print(f"    -> Sau lọc missing: {df_imputed.shape}")
    return df_imputed

def remove_low_variance(df, threshold=1e-8):
    """
    Loại bỏ đặc trưng có phương sai quá thấp (bao gồm toàn 0).
    """
    variances = df.var(axis=0)
    df_filtered = df.loc[:, variances > threshold]
    print(f"    -> Sau lọc low-variance: {df_filtered.shape}")
    return df_filtered

def process_standard_omics(folder_path, omics_name, protein_coding_list=None):
    print(f"\n[BẮT ĐẦU XỬ LÝ {omics_name.upper()}]")
    file_paths = glob.glob(os.path.join(folder_path, "*.tsv"))
    df_list = [clean_and_transpose(file) for file in file_paths]
    print(f"  => Gộp {len(df_list)} file và lấy giao thoa đặc trưng...")
    master_df = pd.concat(df_list, axis=0, join='inner')

    if omics_name.lower() == 'gene' and protein_coding_list is not None:
        master_df.columns = [str(col).split('.')[0] for col in master_df.columns]
        valid_genes = master_df.columns.intersection(protein_coding_list)
        master_df = master_df[valid_genes]
        print(f"  => Còn lại {len(valid_genes)} gene protein-coding.")

    df_imputed = handle_missing_and_impute(master_df)
    df_final = remove_low_variance(df_imputed)
    return df_final

def process_methylation(methyl_folder_path):
    """
    Xử lý methylation: giao thoa chip 27k và 450k.
    """
    print(f"\n[BẮT ĐẦU XỬ LÝ METHYLATION]")
    path_27k = os.path.join(methyl_folder_path, "27k", "*.tsv")
    path_450k = os.path.join(methyl_folder_path, "450k", "*.tsv")
    file_paths = glob.glob(path_27k) + glob.glob(path_450k)
    df_list = [clean_and_transpose(file) for file in file_paths]
    print(f"  => Gộp {len(file_paths)} file (27k & 450k)...")
    master_df = pd.concat(df_list, axis=0, join='inner')
    df_imputed = handle_missing_and_impute(master_df)
    df_final = remove_low_variance(df_imputed)
    return df_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gtf_path', type=str, default=None)
    args = parser.parse_args()

    BASE_DIR = args.input_dir
    OUT_DIR = args.output_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    pc_list = None
    if args.gtf_path:
        pc_list = get_protein_coding_list(args.gtf_path)

    # Gene
    gene_path = os.path.join(BASE_DIR, "gene")
    df_gene = process_standard_omics(gene_path, "Gene", protein_coding_list=pc_list)
    df_gene.to_csv(os.path.join(OUT_DIR, "processed_gene.csv"))

    # miRNA
    mirna_path = os.path.join(BASE_DIR, "mirna")
    df_mirna = process_standard_omics(mirna_path, "miRNA")
    df_mirna.to_csv(os.path.join(OUT_DIR, "processed_mirna.csv"))

    # Methylation
    methyl_path = os.path.join(BASE_DIR, "methyl")
    df_methyl = process_methylation(methyl_path)
    df_methyl.to_csv(os.path.join(OUT_DIR, "processed_methylation.csv"))

    print("\n[HOÀN TẤT!] Toàn bộ ma trận Omics đã được lưu vào thư mục output.")