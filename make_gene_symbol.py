"""
make_gene_symbol.py
===================
Tạo final_gene_symbol.csv từ final_gene.csv bằng cách map
Ensembl ID (ENSG...) → gene symbol dùng HGNC complete set.

File này cần chạy 1 lần cho mỗi dataset (BRCA / UCEC / KIPAN / GI).
Chỉ cần final_gene.csv đã có sẵn trong --final_dir.

Cách chạy trên Colab:
─────────────────────
    # Chạy cho 1 dataset:
    !python make_gene_symbol.py \
        --final_dir  "/kaggle/input/.../data_final_brca" \
        --hgnc_path  "/kaggle/input/.../Heterogeneous_Graph/hgnc_complete_set.txt"

    # Chạy cho nhiều dataset cùng lúc:
    !python make_gene_symbol.py \
        --final_dirs \
            "/kaggle/input/.../data_final_brca" \
            "/kaggle/input/.../data_final_ucec" \
            "/kaggle/input/.../data_final_kipan" \
        --hgnc_path  "/kaggle/input/.../Heterogeneous_Graph/hgnc_complete_set.txt"
"""

import os
import argparse
import pandas as pd


# ─────────────────────────────────────────────
#  HGNC loader (dùng chung cho tất cả datasets)
# ─────────────────────────────────────────────

def load_hgnc_mapping(hgnc_path: str) -> dict:
    """
    Đọc hgnc_complete_set.txt, trả về dict: ensembl_gene_id → symbol.
    Strip version number trước khi lookup (ENSG00000000003.15 → ENSG00000000003).
    """
    print(f"[HGNC] Đọc: {hgnc_path}")
    hgnc = pd.read_csv(
        hgnc_path, sep="\t",
        usecols=["ensembl_gene_id", "symbol"],
        low_memory=False,
        dtype=str,
    )
    hgnc = hgnc.dropna(subset=["ensembl_gene_id", "symbol"])
    hgnc["ensembl_gene_id"] = hgnc["ensembl_gene_id"].str.strip()
    hgnc["symbol"]          = hgnc["symbol"].str.strip()
    # Nếu 1 Ensembl ID có nhiều symbol (rất hiếm) → giữ cái đầu tiên
    hgnc = hgnc.drop_duplicates(subset="ensembl_gene_id", keep="first")
    mapping = dict(zip(hgnc["ensembl_gene_id"], hgnc["symbol"]))
    print(f"[HGNC] {len(mapping):,} Ensembl → symbol mappings")
    return mapping


# ─────────────────────────────────────────────
#  Xử lý 1 dataset
# ─────────────────────────────────────────────

def make_gene_symbol(final_dir: str, ensg_to_symbol: dict) -> None:
    """
    Đọc final_gene.csv từ final_dir, convert cột Ensembl → symbol,
    lưu ra final_gene_symbol.csv trong cùng thư mục.
    """
    gene_path = os.path.join(final_dir, "final_gene.csv")
    out_path  = os.path.join(final_dir, "final_gene_symbol.csv")

    if not os.path.exists(gene_path):
        print(f"[Skip] Không tìm thấy: {gene_path}")
        return

    if os.path.exists(out_path):
        print(f"[Skip] Đã tồn tại: {out_path}  (xóa file nếu muốn tạo lại)")
        return

    print(f"\n{'─'*55}")
    print(f"[Convert] {final_dir}")
    gene = pd.read_csv(gene_path, index_col=0)
    print(f"  Shape gốc : {gene.shape}  (patients × Ensembl IDs)")

    # Map mỗi cột: strip version rồi lookup symbol
    new_cols = []
    for col in gene.columns:
        base   = col.split(".")[0]          # ENSG00000000003.15 → ENSG00000000003
        symbol = ensg_to_symbol.get(base)   # None nếu không tìm thấy
        new_cols.append(symbol)

    # Chỉ giữ cột map được
    mask     = [s is not None for s in new_cols]
    n_mapped = sum(mask)
    print(f"  Map được  : {n_mapped:,}/{gene.shape[1]:,} genes")

    gene_sym         = gene.loc[:, mask].copy()
    gene_sym.columns = [s for s in new_cols if s is not None]

    # Xử lý duplicate symbol (2 ENSG khác nhau → cùng symbol) — giữ cái đầu
    n_before = gene_sym.shape[1]
    gene_sym = gene_sym.loc[:, ~gene_sym.columns.duplicated(keep="first")]
    if gene_sym.shape[1] < n_before:
        print(f"  Loại dup  : {n_before - gene_sym.shape[1]:,} cột trùng symbol")

    gene_sym.to_csv(out_path)
    print(f"  Lưu ra    : {out_path}")
    print(f"  Shape cuối: {gene_sym.shape}  (patients × gene symbols)")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tạo final_gene_symbol.csv từ final_gene.csv + HGNC mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--final_dir", type=str,
        help="Thư mục chứa final_gene.csv (dùng cho 1 dataset)",
    )
    group.add_argument(
        "--final_dirs", type=str, nargs="+",
        help="Nhiều thư mục cùng lúc (dùng cho nhiều dataset)",
    )
    parser.add_argument(
        "--hgnc_path", type=str, required=True,
        help="Đường dẫn hgnc_complete_set.txt (trong Heterogeneous_Graph/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ensg_to_symbol = load_hgnc_mapping(args.hgnc_path)

    dirs = [args.final_dir] if args.final_dir else args.final_dirs
    for d in dirs:
        make_gene_symbol(d, ensg_to_symbol)

    print(f"\nXong! Đã xử lý {len(dirs)} dataset(s).")


if __name__ == "__main__":
    main()
