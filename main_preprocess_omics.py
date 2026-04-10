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

import os
import argparse
import time

import config  # đường dẫn trung tâm, tự phát hiện Colab vs Local

from preprocess_Gene   import process_gene
from preprocess_miRNA  import process_mirna
from preprocess_CpG    import process_cpg
from preprocess_labels import process_clinical_labels
from final_process_omics import final_process


# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline tiền xử lý Multi-Omics TCGA cho MoXGATE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Đường dẫn (default tự động từ config.py) ────────────────────────────
    parser.add_argument(
        "--input_dir", type=str, default=config.GI_RAW_OMICS_DIR,
        help="Thư mục gốc chứa dữ liệu omics (chứa các subfolder gene/, mirna/, methyl/)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=config.GI_PROCESSED_DIR,
        help="Thư mục lưu các file CSV đã xử lý (processed_*.csv và clean_labels.csv)"
    )
    parser.add_argument(
        "--final_dir", type=str, default=config.GI_FINAL_DIR,
        help="Thư mục lưu kết quả cuối (final_*.csv)"
    )
    parser.add_argument(
        "--subtype_dir", type=str, default=config.GI_RAW_SUBTYPE_DIR,
        help="Thư mục chứa các file TSV clinical/subtype"
    )
    parser.add_argument(
        "--gtf_path", type=str, default=config.GTF_PATH,
        help="Đường dẫn file GENCODE GTF annotation (plain hoặc .gz) — dùng cho Gene"
    )

    # ── Đường dẫn phụ trợ cho CpG — tùy chọn ───────────────────────────────
    parser.add_argument(
        "--cross_reactive_path", type=str, default=config.CROSS_REACTIVE_PATH,
        help="(Tùy chọn) File cross-reactive probes (Chen et al. 2013)"
    )
    parser.add_argument(
        "--manifest_path", type=str, default=config.MANIFEST_PATH,
        help="(Tùy chọn) Illumina 450k manifest CSV — dùng để lọc chrX/Y cho CpG"
    )

    # ── Flags skip từng bước (khi file đã có sẵn) ──────────────────────────────
    parser.add_argument("--skip_gene",   action="store_true", help="Bỏ qua bước xử lý Gene")
    parser.add_argument("--skip_mirna",  action="store_true", help="Bỏ qua bước xử lý miRNA")
    parser.add_argument("--skip_cpg",    action="store_true", help="Bỏ qua bước xử lý CpG")
    parser.add_argument("--skip_labels", action="store_true", help="Bỏ qua bước xử lý labels")

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
    print(f"  final_dir  : {args.final_dir}")
    print(f"  subtype_dir: {args.subtype_dir}")
    skipped = [k for k, v in [("gene", args.skip_gene), ("mirna", args.skip_mirna),
                               ("cpg", args.skip_cpg), ("labels", args.skip_labels)] if v]
    if skipped:
        print(f"  skip       : {', '.join(skipped)}")
    print("★"*60)

    total_start = time.time()
    results = {}

    # ── 1. GENE ──────────────────────────────────────────────────────────
    if not args.skip_gene:
        t0 = time.time()
        df = process_gene(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            gtf_path=args.gtf_path,
        )
        results["Gene"] = df.shape
        print(f"  ⏱ Gene xử lý xong trong {time.time()-t0:.1f}s\n")
    else:
        print("  [skip] Gene — dùng processed_gene.csv cũ\n")

    # ── 2. miRNA ──────────────────────────────────────────────────────────
    if not args.skip_mirna:
        t0 = time.time()
        df = process_mirna(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
        results["miRNA"] = df.shape
        print(f"  ⏱ miRNA xử lý xong trong {time.time()-t0:.1f}s\n")
    else:
        print("  [skip] miRNA — dùng processed_mirna.csv cũ\n")

    # ── 3. CpG (Methylation) ──────────────────────────────────────────────
    if not args.skip_cpg:
        t0 = time.time()
        df = process_cpg(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            cross_reactive_path=args.cross_reactive_path,
            manifest_path=args.manifest_path,
        )
        results["CpG"] = df.shape
        print(f"  ⏱ CpG xử lý xong trong {time.time()-t0:.1f}s\n")
    else:
        print("  [skip] CpG — dùng processed_methylation.csv cũ\n")

    # ── 4. LABELS ─────────────────────────────────────────────────────────
    if not args.skip_labels:
        t0 = time.time()
        labels_df = process_clinical_labels(args.subtype_dir)
        os.makedirs(args.output_dir, exist_ok=True)
        labels_path = os.path.join(args.output_dir, "clean_labels.csv")
        labels_df.to_csv(labels_path, index=False)
        results["Labels"] = labels_df.shape
        print(f"  ⏱ Labels xử lý xong trong {time.time()-t0:.1f}s\n")
    else:
        print("  [skip] Labels — dùng clean_labels.csv cũ\n")
        labels_path = os.path.join(args.output_dir, "clean_labels.csv")

    # ── 5. FINAL MERGE ────────────────────────────────────────────────────
    t0 = time.time()
    final_process(
        processed_dir=args.output_dir,
        labels_path=labels_path,
        output_dir=args.final_dir,
    )
    print(f"  ⏱ Final merge xong trong {time.time()-t0:.1f}s\n")

    # ── TỔNG KẾT ─────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    print("\n" + "★"*60)
    print("  HOÀN TẤT! Tổng kết quả:")
    print("─"*60)
    for name, shape in results.items():
        print(f"  {name:8s}: {shape}")
    print(f"\n  ⏱ Tổng thời gian: {elapsed/60:.1f} phút")
    print("★"*60)


if __name__ == "__main__":
    main()