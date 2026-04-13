"""
final_process_omics.py
======================
Bước cuối — lọc 3 file omics đã xử lý, chỉ giữ lại những bệnh nhân
có nhãn trong clean_labels.csv.

Lý do cần bước này:
    - preprocess_Gene/miRNA/CpG xử lý tất cả bệnh nhân trong GDC
    - clean_labels.csv chỉ có bệnh nhân thuộc 5 subtype (CIN/GS/MSI/HM-SNV/EBV)
    - Bệnh nhân không có nhãn → không dùng được cho classification → loại bỏ
    - 3 file omics có số bệnh nhân khác nhau → lấy GIAO (inner join với labels)

Pipeline:
    ┌─────────────────────────────────────────────────────────────────┐
    │ BƯỚC 1 — Đọc clean_labels.csv                                   │
    │   Lấy tập Patient ID có nhãn hợp lệ                             │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 2 — Lọc từng omics                                         │
    │   Chỉ giữ hàng (bệnh nhân) có Patient ID trong tập nhãn         │
    │   → index của omics phải khớp với cột 'Patient ID' của labels   │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 3 — Đồng bộ thứ tự bệnh nhân                               │
    │   Sắp xếp cả 3 omics và labels theo cùng 1 thứ tự Patient ID    │
    │   → đảm bảo hàng i của gene/mirna/cpg tương ứng cùng bệnh nhân  │
    ├─────────────────────────────────────────────────────────────────┤
    │ BƯỚC 4 — Lưu kết quả                                            │
    │   → final_gene.csv                                              │
    │   → final_mirna.csv                                             │
    │   → final_methylation.csv                                       │
    │   → final_labels.csv  (chỉ những bệnh nhân có đủ cả 3 omics)    │
    └─────────────────────────────────────────────────────────────────┘

Cách chạy:
    !python final_process_omics.py \\
        --processed_dir "/content/drive/MyDrive/ĐATN_2025.2/data_processed" \\
        --labels_path   "/content/drive/MyDrive/ĐATN_2025.2/data_processed/clean_labels.csv" \\
        --output_dir    "/content/drive/MyDrive/ĐATN_2025.2/data_final"
"""

import os
import argparse
import pandas as pd


# ─────────────────────────────────────────────
# 1. ĐỌC LABELS
# ─────────────────────────────────────────────

def load_labels(labels_path: str) -> pd.DataFrame:
    """
    Đọc file clean_labels.csv, trả về DataFrame với index là Patient ID.

    Args:
        labels_path: Đường dẫn file clean_labels.csv.

    Returns:
        DataFrame với index = Patient ID, các cột: Cancer_Type, Clean_Subtype, Target_Label.
    """
    print(f"[Final] Đọc labels: {os.path.basename(labels_path)}")
    labels = pd.read_csv(labels_path, index_col=0)

    # index có thể là Patient ID hoặc cột số — chuẩn hoá về Patient ID
    # Nếu index là số (0,1,2,...) thì Patient ID là cột riêng
    if not labels.index.astype(str).str.startswith('TCGA').any():
        if 'Patient ID' in labels.columns:
            labels = labels.set_index('Patient ID')
        else:
            raise ValueError(
                "Không tìm thấy cột 'Patient ID' trong labels.\n"
                "Kiểm tra lại file clean_labels.csv."
            )

    print(f"[Final] ✓ {len(labels):,} bệnh nhân có nhãn hợp lệ")
    print(f"[Final]   Phân bố subtype:\n{labels['Target_Label'].value_counts().sort_index().to_string()}")
    return labels


# ─────────────────────────────────────────────
# 2. LỌC OMICS THEO LABELS
# ─────────────────────────────────────────────

def filter_by_labels(omics_path: str, labeled_patients: pd.Index, name: str) -> pd.DataFrame:
    """
    Đọc 1 file omics đã xử lý, chỉ giữ bệnh nhân có trong labeled_patients.

    Args:
        omics_path:       Đường dẫn file CSV omics (index = Patient ID).
        labeled_patients: Index chứa Patient ID có nhãn hợp lệ.
        name:             Tên omics để in log (Gene/miRNA/Methylation).

    Returns:
        DataFrame (n_labeled_patients, n_features), index = Patient ID.
    """
    print(f"\n[Final] Đọc {name}: {os.path.basename(omics_path)}")
    df = pd.read_csv(omics_path, index_col=0)
    print(f"[Final]   Shape gốc: {df.shape} (Bệnh nhân × Features)")

    # Kiểm tra Patient ID trùng lặp
    dup_mask = df.index.duplicated(keep=False)
    if dup_mask.any():
        dup_ids = df.index[dup_mask].unique().tolist()
        for pid in dup_ids:
            rows = df.loc[pid]
            if rows.duplicated().any() or (rows.iloc[0] == rows.iloc[1]).all():
                # 2 hàng giống hệt nhau → bỏ 1 không sao
                print(f"[Final]   ⚠ '{pid}' xuất hiện 2 lần nhưng dữ liệu GIỐNG NHAU → bỏ bản sao")
            else:
                # 2 hàng khác nhau → cần điều tra trong file preprocess
                print(f"[Final]   ✗ '{pid}' xuất hiện 2 lần với dữ liệu KHÁC NHAU → giữ lần đầu (cần kiểm tra {name}!)")
        df = df[~df.index.duplicated(keep='first')]

    # Lấy giao: bệnh nhân vừa có trong omics vừa có nhãn
    common = df.index.intersection(labeled_patients)
    df_filtered = df.loc[common]

    missing = labeled_patients.difference(df.index)
    if len(missing) > 0:
        print(f"[Final]   ⚠ {len(missing):,} bệnh nhân có nhãn nhưng không có trong {name} → bỏ qua:")
        for pid in sorted(missing):
            print(f"[Final]       - {pid}")

    print(f"[Final]   Shape sau lọc: {df_filtered.shape}")
    return df_filtered


# ─────────────────────────────────────────────
# 3. ĐỒNG BỘ THỨ TỰ BỆNH NHÂN
# ─────────────────────────────────────────────

def align_patients(*dfs: pd.DataFrame) -> list:
    """
    Lấy giao của tất cả Patient ID trong các DataFrame,
    sắp xếp theo thứ tự chung → đảm bảo hàng i luôn là cùng bệnh nhân.

    Args:
        *dfs: Các DataFrame omics, mỗi cái có index = Patient ID.

    Returns:
        List các DataFrame đã được align theo cùng thứ tự Patient ID.
    """
    # Lấy giao của tất cả index
    common_patients = dfs[0].index
    for df in dfs[1:]:
        common_patients = common_patients.intersection(df.index)

    common_patients = common_patients.sort_values()

    # In các bệnh nhân bị loại ở bước giao (có trong labels nhưng thiếu ≥1 omics)
    labels_df = dfs[-1]  # labels luôn là phần tử cuối
    dropped = labels_df.index.difference(common_patients)
    if len(dropped) > 0:
        print(f"\n[Final] ⚠ {len(dropped):,} bệnh nhân bị loại do thiếu ít nhất 1 omics:")
        for pid in sorted(dropped):
            cancer = labels_df.loc[pid, 'Cancer_Type'] if 'Cancer_Type' in labels_df.columns else '?'
            subtype = labels_df.loc[pid, 'Clean_Subtype'] if 'Clean_Subtype' in labels_df.columns else '?'
            print(f"[Final]   - {pid}  [{cancer} / {subtype}]")

    print(f"\n[Final] Bệnh nhân chung (có đủ cả 3 omics + nhãn): {len(common_patients):,}")

    return [df.loc[common_patients] for df in dfs]


# ─────────────────────────────────────────────
# 4. HÀM CHÍNH
# ─────────────────────────────────────────────

def final_process(processed_dir: str, labels_path: str, output_dir: str):
    """
    Pipeline cuối: lọc 3 omics theo labels, đồng bộ thứ tự, lưu kết quả.

    Args:
        processed_dir: Thư mục chứa processed_gene.csv, processed_mirna.csv,
                       processed_methylation.csv.
        labels_path:   Đường dẫn file clean_labels.csv.
        output_dir:    Thư mục lưu các file final_*.csv.
    """
    print("\n" + "="*60)
    print("  BẮT ĐẦU BƯỚC CUỐI: LỌC BỆNH NHÂN CÓ NHÃN")
    print("="*60)

    # ── Bước 1: Đọc labels ──────────────────────────────────────────────
    labels = load_labels(labels_path)

    # ── Bước 2: Lọc từng omics theo labels ─────────────────────────────
    gene  = filter_by_labels(
        os.path.join(processed_dir, "processed_gene.csv"),
        labels.index, "Gene"
    )
    mirna = filter_by_labels(
        os.path.join(processed_dir, "processed_mirna.csv"),
        labels.index, "miRNA"
    )
    methyl = filter_by_labels(
        os.path.join(processed_dir, "processed_methylation.csv"),
        labels.index, "Methylation"
    )

    # ── Bước 3: Đồng bộ thứ tự bệnh nhân ──────────────────────────────
    # Lấy giao cả 3 omics + labels → bệnh nhân có đầy đủ tất cả
    gene, mirna, methyl, labels_final = align_patients(gene, mirna, methyl, labels)

    print(f"\n[Final] Phân bố subtype cuối cùng:")
    print(labels_final['Target_Label'].value_counts().sort_index().to_string())
    if 'Clean_Subtype' in labels_final.columns:
        print(labels_final['Clean_Subtype'].value_counts().to_string())

    # Thống kê theo Cancer_Type (quan trọng: ESCA dùng để test theo paper)
    if 'Cancer_Type' in labels_final.columns:
        print(f"\n[Final] Thống kê theo Cancer_Type:")
        if 'Clean_Subtype' in labels_final.columns:
            ct_stats = labels_final.groupby('Cancer_Type')['Clean_Subtype'].value_counts().unstack(fill_value=0)
            print(ct_stats.to_string())
        else:
            # KIPAN / datasets không có Clean_Subtype: chỉ in số lượng theo Cancer_Type
            ct_stats = labels_final.groupby('Cancer_Type')['Target_Label'].count()
            print(ct_stats.to_string())
        print(f"\n[Final] Tổng bệnh nhân mỗi loại ung thư:")
        print(labels_final['Cancer_Type'].value_counts().to_string())
        if labels_final['Cancer_Type'].isin(['ESCA']).any():
            print(f"\n[Final] ℹ️  Theo paper: ESCA ({labels_final[labels_final['Cancer_Type']=='ESCA'].shape[0]} bệnh nhân) dùng để test,"
                  f" COAD+READ+STAD ({labels_final[labels_final['Cancer_Type']!='ESCA'].shape[0]} bệnh nhân) dùng để train.")

    # ── Bước 4: Lưu kết quả ─────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    out_gene   = os.path.join(output_dir, "final_gene.csv")
    out_mirna  = os.path.join(output_dir, "final_mirna.csv")
    out_methyl = os.path.join(output_dir, "final_methylation.csv")
    out_labels = os.path.join(output_dir, "final_labels.csv")

    gene.to_csv(out_gene)
    mirna.to_csv(out_mirna)
    methyl.to_csv(out_methyl)
    labels_final.to_csv(out_labels)

    print(f"\n[Final] ✓ Đã lưu:")
    print(f"         {out_gene}        → {gene.shape}")
    print(f"         {out_mirna}       → {mirna.shape}")
    print(f"         {out_methyl}  → {methyl.shape}")
    print(f"         {out_labels}      → {labels_final.shape}")
    print("="*60)


# ─────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bước cuối: lọc 3 omics theo bệnh nhân có nhãn"
    )
    parser.add_argument(
        "--processed_dir", type=str, required=True,
        help="Thư mục chứa processed_gene.csv, processed_mirna.csv, processed_methylation.csv"
    )
    parser.add_argument(
        "--labels_path", type=str, required=True,
        help="Đường dẫn file clean_labels.csv"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Thư mục lưu final_gene.csv, final_mirna.csv, final_methylation.csv, final_labels.csv"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    final_process(
        processed_dir=args.processed_dir,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
    )
