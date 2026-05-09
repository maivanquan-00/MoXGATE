"""
main_preprocess_omics.py
========================
Pipeline tiền xử lý multi-omics TCGA cho GIAC — chạy 1 lệnh cho tất cả datasets.

Cách dùng (Colab sau khi mount Drive):
─────────────────────────────────────────────────────────────────────
    # Build TẤT CẢ datasets từ raw → final (với gene_symbol):
    !python preprocessed_data/main_preprocess_omics.py \\
        --base /content/drive/MyDrive/ĐATN_2025.2

    # Chỉ rebuild GENE cho cả 4 dataset (use case sau khi fix double-log):
    !python preprocessed_data/main_preprocess_omics.py \\
        --base /content/drive/MyDrive/ĐATN_2025.2 \\
        --only gene

    # Chỉ 1 dataset:
    !python preprocessed_data/main_preprocess_omics.py \\
        --base /content/drive/MyDrive/ĐATN_2025.2 \\
        --dataset brca

    # Rebuild labels GI (default: SKIP labels — chỉ GI mới được hỗ trợ):
    !python preprocessed_data/main_preprocess_omics.py \\
        --base /content/drive/MyDrive/ĐATN_2025.2 \\
        --dataset gi --rebuild_labels_gi

Cấu trúc thư mục cần có dưới --base:
─────────────────────────────────────────────────────────────────────
    <base>/
    ├── data_original/
    │   ├── multi_omics/         (GI: COAD/STAD/READ/ESCA, có sub gene/, mirna/, methyl/)
    │   ├── multi_omics_brca/
    │   ├── multi_omics_ucec/
    │   ├── multi_omics_kipan/
    │   ├── multi_omics_lgg/     (TCGA-LGG.gene.tsv, .mirna.tsv, methylation 450k)
    │   ├── subtype/             (TSV labels GI — chỉ GI chạy được pipeline labels)
    │   └── annotation/
    │       ├── gencode.v36.annotation.gtf  (bắt buộc)
    │       ├── cross_reactive_probes.txt   (tùy chọn, cho CpG)
    │       └── HumanMethylation450_manifest.csv (tùy chọn, cho CpG)
    ├── data_processed*/         (intermediate, tự tạo)
    ├── data_final*/             (output cuối, tự tạo)
    └── Heterogeneous_Graph/
        └── hgnc_complete_set.txt  (cho gene_symbol mapping)

⚠️  LƯU Ý LABELS — pipeline labels CHỈ tự build cho GI:
    - GI    : --rebuild_labels_gi → tự build từ data_original/subtype/*.tsv
    - BRCA, UCEC, KIPAN, LGG: bạn phải tự tạo file
        data_processed_<dataset>/clean_labels_<dataset>.csv
      với cột: Patient ID, Cancer_Type (optional), Clean_Subtype, Target_Label
      Format: Target_Label = integer 0..(num_classes-1)
      LGG mapping đề xuất: {Codel: 0, IDHmut-non-codel: 1, IDHwt: 2}

Output mỗi dataset (sau khi chạy xong):
    data_final*/
    ├── final_gene.csv          (gene symbol cols, log2(norm_count+1) từ Xena)
    │                           (đã map ENSG→symbol bên trong preprocess_Gene)
    ├── final_mirna.csv         (log2(RPM+1) từ Xena)
    ├── final_methylation.csv   (beta values [0,1])
    └── final_labels.csv        (Patient ID + Cancer_Type + Target_Label)
"""

import os
import argparse
import sys
import time

import pandas as pd

# Đảm bảo thư mục chứa file này nằm trong sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess_Gene    import process_gene
from preprocess_miRNA   import process_mirna
from preprocess_CpG     import process_cpg
from preprocess_labels  import process_clinical_labels
from final_process_omics import final_process


# ─────────────────────────────────────────────────────────────────────
#  Registry — paths cho 4 datasets, relative to --base
# ─────────────────────────────────────────────────────────────────────

DATASETS = {
    "gi": {
        "raw":          "data_original/multi_omics",
        "processed":    "data_processed",
        "final":        "data_final",
        "subtype":      "data_original/subtype",
        "labels_file":  "clean_labels.csv",
    },
    "brca": {
        "raw":          "data_original/multi_omics_brca",
        "processed":    "data_processed_brca",
        "final":        "data_final_brca",
        "subtype":      "data_original/subtype_brca",
        "labels_file":  "clean_labels_brca.csv",
    },
    "ucec": {
        "raw":          "data_original/multi_omics_ucec",
        "processed":    "data_processed_ucec",
        "final":        "data_final_ucec",
        "subtype":      "data_original/subtype_ucec",
        "labels_file":  "clean_labels_ucec.csv",
    },
    "kipan": {
        "raw":          "data_original/multi_omics_kipan",
        "processed":    "data_processed_kipan",
        "final":        "data_final_kipan",
        "subtype":      "data_original/subtype_kipan",
        "labels_file":  "clean_labels_kipan.csv",
    },
    "lgg": {
        "raw":          "data_original/multi_omics_lgg",
        "processed":    "data_processed_lgg",
        "final":        "data_final_lgg",
        "subtype":      "data_original/subtype_lgg",
        "labels_file":  "clean_labels_lgg.csv",
    },
}


# ─────────────────────────────────────────────────────────────────────
#  Argument parser
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline preprocess multi-omics TCGA — 1 lệnh cho tất cả datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base", type=str, required=True,
                   help="Base directory chứa data_original/, data_final*/, Heterogeneous_Graph/")
    p.add_argument("--dataset", type=str, default="all",
                   choices=list(DATASETS.keys()) + ["all"],
                   help="Dataset cần process. 'all' = chạy hết 5 datasets (gi/brca/ucec/kipan/lgg)")
    p.add_argument("--only", type=str, default=None,
                   choices=["gene", "mirna", "cpg"],
                   help="Chỉ chạy 1 omic (vẫn chạy final_merge + gene_symbol). Mặc định: tất cả")
    p.add_argument("--rebuild_labels_gi", action="store_true",
                   help="Rebuild clean_labels.csv cho GI (chỉ work với GI). Mặc định: skip labels.")

    # File phụ trợ — auto detect từ --base nếu không pass
    p.add_argument("--gtf_path", type=str, default=None,
                   help="Auto: <base>/data_original/annotation/gencode.v36.annotation.gtf")
    p.add_argument("--cross_reactive_path", type=str, default=None,
                   help="Auto: <base>/data_original/annotation/cross_reactive_probes.txt")
    p.add_argument("--manifest_path", type=str, default=None,
                   help="Auto: <base>/data_original/annotation/HumanMethylation450_manifest.csv")
    p.add_argument("--hgnc_path", type=str, default=None,
                   help="Auto: <base>/Heterogeneous_Graph/hgnc_complete_set.txt")

    args = p.parse_args()

    # Auto-fill paths
    if args.gtf_path is None:
        args.gtf_path = os.path.join(args.base, "data_original/annotation/gencode.v36.annotation.gtf")
    if args.cross_reactive_path is None:
        args.cross_reactive_path = os.path.join(args.base, "data_original/annotation/cross_reactive_probes.txt")
    if args.manifest_path is None:
        args.manifest_path = os.path.join(args.base, "data_original/annotation/HumanMethylation450_manifest.csv")
    if args.hgnc_path is None:
        args.hgnc_path = os.path.join(args.base, "Heterogeneous_Graph/hgnc_complete_set.txt")

    return args


# ─────────────────────────────────────────────────────────────────────
#  Process 1 dataset
# ─────────────────────────────────────────────────────────────────────

def process_one_dataset(name: str, args) -> dict:
    cfg = DATASETS[name]
    raw_dir       = os.path.join(args.base, cfg["raw"])
    processed_dir = os.path.join(args.base, cfg["processed"])
    final_dir     = os.path.join(args.base, cfg["final"])
    subtype_dir   = os.path.join(args.base, cfg["subtype"])
    labels_path   = os.path.join(processed_dir, cfg["labels_file"])

    print("\n" + "█" * 70)
    print(f"  DATASET: {name.upper()}")
    print("█" * 70)
    print(f"  Raw       : {raw_dir}")
    print(f"  Processed : {processed_dir}")
    print(f"  Final     : {final_dir}")

    if not os.path.isdir(raw_dir):
        print(f"  ⚠️  Raw dir không tồn tại — bỏ qua {name}")
        return {}

    do_gene  = args.only is None or args.only == "gene"
    do_mirna = args.only is None or args.only == "mirna"
    do_cpg   = args.only is None or args.only == "cpg"
    do_labels = (name == "gi") and args.rebuild_labels_gi

    timings = {}

    # ── 1. GENE (kèm map Ensembl → symbol nếu có HGNC) ──────────────
    if do_gene:
        t0 = time.time()
        process_gene(
            input_dir  = raw_dir,
            output_dir = processed_dir,
            gtf_path   = args.gtf_path,
            hgnc_path  = args.hgnc_path,
        )
        timings["gene"] = time.time() - t0
    else:
        print(f"\n  [skip gene]")

    # ── 2. miRNA ────────────────────────────────────────────────────
    if do_mirna:
        t0 = time.time()
        process_mirna(
            input_dir  = raw_dir,
            output_dir = processed_dir,
        )
        timings["mirna"] = time.time() - t0
    else:
        print(f"\n  [skip mirna]")

    # ── 3. CpG ──────────────────────────────────────────────────────
    if do_cpg:
        t0 = time.time()
        process_cpg(
            input_dir          = raw_dir,
            output_dir         = processed_dir,
            cross_reactive_path = args.cross_reactive_path,
            manifest_path      = args.manifest_path,
        )
        timings["cpg"] = time.time() - t0
    else:
        print(f"\n  [skip cpg]")

    # ── 4. Labels (CHỈ GI) ──────────────────────────────────────────
    if do_labels:
        t0 = time.time()
        labels_df = process_clinical_labels(subtype_dir)
        os.makedirs(processed_dir, exist_ok=True)
        labels_df.to_csv(labels_path, index=False)
        print(f"  ✓ Labels saved: {labels_path}")
        timings["labels"] = time.time() - t0
    else:
        if name == "gi" and not os.path.exists(labels_path):
            print(f"\n  ⚠️  Labels GI chưa có và --rebuild_labels_gi không bật — final_merge sẽ fail")
        elif name in ("brca", "ucec", "kipan", "lgg") and not os.path.exists(labels_path):
            print(f"\n  ⚠️  {labels_path} chưa có — pipeline không tự build labels cho {name.upper()}.")
            print(f"     Bạn cần tự tạo file này (xem docstring đầu file). final_merge sẽ skip.")

    # ── 5. Final merge ──────────────────────────────────────────────
    if not os.path.exists(labels_path):
        print(f"\n  ⚠️  {labels_path} không tồn tại — bỏ qua final_merge cho {name}")
        return timings

    t0 = time.time()
    final_process(
        processed_dir = processed_dir,
        labels_path   = labels_path,
        output_dir    = final_dir,
    )
    timings["final_merge"] = time.time() - t0

    # ── Cleanup legacy: xóa final_gene_symbol.csv cũ nếu có ─────────
    # Pipeline mới gộp Ensembl→symbol vào preprocess_Gene → final_gene.csv đã là symbol.
    # File final_gene_symbol.csv là legacy, không còn cần thiết.
    legacy = os.path.join(final_dir, "final_gene_symbol.csv")
    if os.path.exists(legacy):
        os.remove(legacy)
        print(f"  [cleanup] Đã xóa legacy: {legacy}")

    return timings


# ─────────────────────────────────────────────────────────────────────
#  Sanity check final files — phát hiện double-log bug & co.
# ─────────────────────────────────────────────────────────────────────

def sanity_check_dataset(name: str, args) -> list:
    """Trả về list các issue phát hiện (rỗng nếu OK)."""
    cfg = DATASETS[name]
    final_dir = os.path.join(args.base, cfg["final"])
    issues = []

    if not os.path.isdir(final_dir):
        return [(name, "MISSING", "Final dir không tồn tại")]

    files = {
        "gene":   "final_gene.csv",
        "mirna":  "final_mirna.csv",
        "meth":   "final_methylation.csv",
        "labels": "final_labels.csv",
    }

    print(f"\n  [{name.upper()}]")
    for omic, fname in files.items():
        path = os.path.join(final_dir, fname)
        if not os.path.exists(path):
            print(f"    ✗ {fname:<28} KHÔNG TỒN TẠI")
            issues.append((name, omic, "missing file"))
            continue

        df = pd.read_csv(path, index_col=0, low_memory=False)

        if omic == "labels":
            print(f"    ✓ {fname:<28} shape={df.shape}")
            if "Target_Label" in df.columns:
                dist = df["Target_Label"].value_counts().sort_index().to_dict()
                print(f"      Phân bố Target_Label: {dist}")
            continue

        vals = df.values
        s = pd.Series(vals.flatten())
        vmin, vmax, vmed = float(s.min()), float(s.max()), float(s.median())
        n_nan = int(pd.isna(vals).sum())

        # Phát hiện anomaly
        verdict = "✓"
        if omic == "gene":
            if vmax < 6.0:
                verdict = "🚨 DOUBLE-LOG"
                issues.append((name, omic, f"max={vmax:.2f} → double-log bug"))
            elif vmax > 30.0:
                verdict = "⚠ unlogged?"
                issues.append((name, omic, f"max={vmax:.2f} → có thể chưa log"))
            # Check cột phải là gene symbol, không phải Ensembl
            cols = df.columns[:5].tolist()
            if any(isinstance(c, str) and c.startswith("ENSG") for c in cols):
                verdict += " ⚠ Ensembl"
                issues.append((name, omic, "cột vẫn Ensembl ID — graph PPI/Reactome sẽ rỗng. Check HGNC path."))
        elif omic == "mirna":
            if vmax < 6.0:
                verdict = "🚨 DOUBLE-LOG?"
                issues.append((name, omic, f"max={vmax:.2f} → có thể double-log"))
            elif vmax > 100.0:
                verdict = "🚨 NO-LOG"
                issues.append((name, omic, f"max={vmax:.2f} → thiếu log transform"))
        elif omic == "meth":
            if not (0 <= vmin and vmax <= 1.05):
                verdict = "⚠ not-beta"
                issues.append((name, omic, f"range [{vmin:.3f},{vmax:.3f}] → không phải beta"))

        if n_nan > 0:
            verdict += f"  +{n_nan} NaN"
            issues.append((name, omic, f"{n_nan} NaN sót"))

        print(f"    {verdict:<18} {fname:<28} shape={df.shape}  range=[{vmin:.3f}, {vmax:.3f}]  median={vmed:.3f}")

    return issues


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not os.path.isdir(args.base):
        print(f"❌ Base dir không tồn tại: {args.base}")
        sys.exit(1)

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    # Header
    print("\n" + "★" * 70)
    print("  MULTI-OMICS PREPROCESSING — GIAC")
    print("★" * 70)
    print(f"  Base       : {args.base}")
    print(f"  Datasets   : {targets}")
    print(f"  Mode       : {args.only or 'full (gene+mirna+cpg)'}")
    print(f"  Skip labels: {'NO (rebuild)' if args.rebuild_labels_gi else 'YES'}")
    print(f"  GTF        : {args.gtf_path}")
    print(f"  HGNC       : {args.hgnc_path}")
    print("★" * 70)

    # Verify HGNC file (sẽ được load lại trong mỗi process_gene call)
    if not os.path.exists(args.hgnc_path):
        print(f"  ⚠️  HGNC không tồn tại tại: {args.hgnc_path}")
        print(f"      → cột gene sẽ giữ Ensembl ID, graph PPI/Reactome RỖNG.")

    # Process từng dataset
    total_start = time.time()
    all_timings = {}
    for name in targets:
        all_timings[name] = process_one_dataset(name, args)

    # ── Sanity check toàn bộ ─────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  SANITY CHECK — final files")
    print("█" * 70)
    all_issues = []
    for name in targets:
        all_issues.extend(sanity_check_dataset(name, args))

    # Summary
    elapsed = time.time() - total_start
    print("\n" + "★" * 70)
    print(f"  HOÀN TẤT — tổng thời gian: {elapsed/60:.1f} phút")
    print("★" * 70)

    if not all_issues:
        print("  ✓ Tất cả datasets pass sanity check — sẵn sàng upload Kaggle")
    else:
        print(f"  ⚠️  Phát hiện {len(all_issues)} vấn đề cần xử lý:")
        for name, omic, msg in all_issues:
            print(f"       [{name.upper()}/{omic}] {msg}")

    # In timing
    print(f"\n  Timing per dataset:")
    for name, ts in all_timings.items():
        if ts:
            total = sum(ts.values())
            steps = "  ".join(f"{k}={v:.0f}s" for k, v in ts.items())
            print(f"    {name.upper():<6} total={total/60:.1f}min  ({steps})")


if __name__ == "__main__":
    main()
