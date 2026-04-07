"""
main_preprocess_omics.py
========================
File điều phối chính — gọi 3 pipeline xử lý omics:
    1. preprocess_Gene.py   → processed_gene.csv
    2. preprocess_miRNA.py  → processed_mirna.csv
    3. preprocess_CpG.py    → processed_methylation.csv

Cách chạy trên Colab:
─────────────────────
    # Clone repo về Colab
    !git clone https://github.com/maivanquan-00/MoXGATE
    %cd MoXGATE

    # Mount Google Drive (nếu chưa mount)
    from google.colab import drive
    drive.mount('/content/drive')

    # Chạy toàn bộ pipeline
    !python main_preprocess_omics.py \\
        --input_dir  "/content/drive/MyDrive/ĐATN_2025.2/data_original/multi_omics" \\
        --output_dir "/content/drive/MyDrive/ĐATN_2025.2/data_processed" \\
        --gtf_path   "/content/drive/MyDrive/ĐATN_2025.2/data_original/annotation/gencode.v36.annotation.gtf"

    # Nếu có file cross-reactive probes và manifest (để lọc methylation chính xác hơn):
    !python main_preprocess_omics.py \\
        --input_dir              "/content/drive/MyDrive/ĐATN_2025.2/data_original/multi_omics" \\
        --output_dir             "/content/drive/MyDrive/ĐATN_2025.2/data_processed" \\
        --gtf_path               "/content/drive/MyDrive/ĐATN_2025.2/data_original/annotation/gencode.v36.annotation.gtf" \\
        --cross_reactive_path    "/content/drive/MyDrive/ĐATN_2025.2/data_original/annotation/cross_reactive_probes.txt" \\
        --manifest_path          "/content/drive/MyDrive/ĐATN_2025.2/data_original/annotation/HumanMethylation450_manifest.csv"

    # Chạy riêng từng omics:
    !python main_preprocess_omics.py --only gene  ...args...
    !python main_preprocess_omics.py --only mirna ...args...
    !python main_preprocess_omics.py --only cpg   ...args...

Cấu trúc thư mục dữ liệu:
──────────────────────────
    data_original/
    ├── multi_omics/
    │   ├── gene/
    │   │   ├── TCGA-COAD.gene.tsv
    │   │   ├── TCGA-ESCA.gene.tsv
    │   │   ├── TCGA-READ.gene.tsv
    │   │   └── TCGA-STAD.gene.tsv
    │   ├── mirna/
    │   │   ├── TCGA-COAD.mirna.tsv
    │   │   ├── TCGA-ESCA.mirna.tsv
    │   │   ├── TCGA-READ.mirna.tsv
    │   │   └── TCGA-STAD.mirna.tsv
    │   └── methyl/
    │       ├── 27k/
    │       │   ├── TCGA-COAD.methylation27.tsv
    │       │   ├── TCGA-READ.methylation27.tsv
    │       │   └── TCGA-STAD.methylation27.tsv
    │       └── 450k/
    │           ├── TCGA-COAD.methylation450.tsv
    │           ├── TCGA-ESCA.methylation450.tsv
    │           ├── TCGA-READ.methylation450.tsv
    │           └── TCGA-STAD.methylation450.tsv
    └── annotation/
        ├── gencode.v36.annotation.gtf        ← bắt buộc (plain hoặc .gz)
        ├── cross_reactive_probes.txt          ← tùy chọn
        └── HumanMethylation450_manifest.csv   ← tùy chọn

Kết quả kỳ vọng:
────────────────
    processed_gene.csv        → ~(1220, 20530)
    processed_mirna.csv       → ~(1225,   746)
    processed_methylation.csv → ~(1255, 23381)
"""

import argparse
import time

from preprocess_Gene  import process_gene
from preprocess_miRNA import process_mirna
from preprocess_CpG   import process_cpg


# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline tiền xử lý Multi-Omics TCGA cho MoXGATE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Đường dẫn cơ bản — bắt buộc
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Thư mục gốc chứa dữ liệu omics (chứa các subfolder gene/, mirna/, methyl/)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Thư mục lưu các file CSV đã xử lý"
    )
    parser.add_argument(
        "--gtf_path", type=str, required=True,
        help="Đường dẫn file GENCODE GTF annotation (plain hoặc .gz) — dùng cho Gene"
    )

    # Đường dẫn phụ trợ cho CpG — tùy chọn
    parser.add_argument(
        "--cross_reactive_path", type=str, default=None,
        help="(Tùy chọn) File cross-reactive probes (Chen et al. 2013) — dùng cho CpG"
    )
    parser.add_argument(
        "--manifest_path", type=str, default=None,
        help="(Tùy chọn) Illumina 450k manifest CSV — dùng để lọc chrX/Y cho CpG"
    )

    # Chạy riêng từng omics
    parser.add_argument(
        "--only", type=str, default=None,
        choices=["gene", "mirna", "cpg"],
        help="Chỉ chạy 1 omics cụ thể. Mặc định: chạy cả 3"
    )

    return parser.parse_args()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "★"*60)
    print("  MoXGATE — MULTI-OMICS PREPROCESSING PIPELINE")
    print("★"*60)
    print(f"  input_dir  : {args.input_dir}")
    print(f"  output_dir : {args.output_dir}")
    print(f"  gtf_path   : {args.gtf_path}")
    if args.cross_reactive_path:
        print(f"  cross_react: {args.cross_reactive_path}")
    if args.manifest_path:
        print(f"  manifest   : {args.manifest_path}")
    if args.only:
        print(f"  only       : {args.only}")
    print("★"*60)

    total_start = time.time()
    results = {}

    run_all = args.only is None

    # ── 1. GENE ──────────────────────────────
    if run_all or args.only == "gene":
        t0 = time.time()
        df_gene = process_gene(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            gtf_path=args.gtf_path,
        )
        results["gene"] = df_gene.shape
        print(f"  ⏱ Gene xử lý xong trong {time.time()-t0:.1f}s\n")

    # ── 2. miRNA ─────────────────────────────
    if run_all or args.only == "mirna":
        t0 = time.time()
        df_mirna = process_mirna(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
        results["mirna"] = df_mirna.shape
        print(f"  ⏱ miRNA xử lý xong trong {time.time()-t0:.1f}s\n")

    # ── 3. CpG (Methylation) ─────────────────
    if run_all or args.only == "cpg":
        t0 = time.time()
        df_cpg = process_cpg(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            cross_reactive_path=args.cross_reactive_path,
            manifest_path=args.manifest_path,
        )
        results["cpg"] = df_cpg.shape
        print(f"  ⏱ CpG xử lý xong trong {time.time()-t0:.1f}s\n")

    # ── TỔNG KẾT ─────────────────────────────
    elapsed = time.time() - total_start
    print("\n" + "★"*60)
    print("  HOÀN TẤT! Tổng kết quả:")
    print("─"*60)

    expected = {
        "gene":  "(~1220, ~20530)",
        "mirna": "(~1225,   ~746)",
        "cpg":   "(~1255, ~23381)",
    }
    label = {
        "gene":  "Gene Expression   → processed_gene.csv",
        "mirna": "miRNA Expression  → processed_mirna.csv",
        "cpg":   "DNA Methylation   → processed_methylation.csv",
    }

    for key, shape in results.items():
        print(f"  {label[key]}")
        print(f"      Shape thực tế : {shape}")
        print(f"      Kỳ vọng paper : {expected[key]}")
        print()

    print(f"  ⏱ Tổng thời gian: {elapsed/60:.1f} phút")
    print("★"*60)


if __name__ == "__main__":
    main()