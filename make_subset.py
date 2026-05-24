"""
make_subset.py
==============
Tạo subset dataset từ thư mục data_final đã có, để chạy baseline MoXGATE
trên các biến thể khác nhau (lọc class hoặc lọc cancer type).

Chiến lược:
    - Đọc 4 file final_*.csv (gene, mirna, methylation, labels)
    - Lọc theo Cancer_Type và/hoặc Subtype
    - Re-index Target_Label về 0..K-1 (contiguous) để khớp với num_classes trong model
    - Lưu ra thư mục mới với cùng 4 file → chạy train_kfold.py như bình thường

Các use case của dự án:
    # GIAC bỏ HM-SNV (898 samples, 4 classes)
    python make_subset.py --input data_final --output data_final_gi_no_hmsnv \
        --exclude_subtypes HM-SNV

    # BRCA bỏ Normal (929 samples, 4 classes)
    python make_subset.py --input data_final_brca --output data_final_brca_no_normal \
        --exclude_subtypes Normal

    # STAD only — 5 classes (380 samples)
    python make_subset.py --input data_final --output data_final_stad \
        --include_cancers STAD

    # STAD only — 4 classes, bỏ HM-SNV (373 samples)
    python make_subset.py --input data_final --output data_final_stad_no_hmsnv \
        --include_cancers STAD --exclude_subtypes HM-SNV

Sau đó chạy baseline với train_kfold.py (tự detect num_classes):
    python train_kfold.py --data_dir data_final_gi_no_hmsnv      --save_path results_gi_no_hmsnv.json
    python train_kfold.py --data_dir data_final_brca_no_normal   --save_path results_brca_no_normal.json
    python train_kfold.py --data_dir data_final_stad             --save_path results_stad.json
    python train_kfold.py --data_dir data_final_stad_no_hmsnv    --save_path results_stad_no_hmsnv.json
"""

import os
import sys
import argparse
import pandas as pd

# Đảm bảo stdout dùng UTF-8 trên Windows (tránh lỗi cp1252 với ký tự tiếng Việt)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def make_subset(
    input_dir: str,
    output_dir: str,
    include_cancers: list[str] | None = None,
    exclude_subtypes: list[str] | None = None,
):
    """
    Lọc subset từ thư mục data_final đã có và lưu ra thư mục mới.

    Args:
        input_dir:        Thư mục chứa final_*.csv (gốc).
        output_dir:       Thư mục lưu output (sẽ tạo nếu chưa có).
        include_cancers:  List Cancer_Type cần giữ (vd ['STAD']). None = giữ tất cả.
        exclude_subtypes: List Subtype cần loại (vd ['HM-SNV']). None = giữ tất cả.
    """
    print("\n" + "=" * 70)
    print(f"  MAKE SUBSET")
    print("=" * 70)
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")
    if include_cancers:
        print(f"  Include cancers  : {include_cancers}")
    if exclude_subtypes:
        print(f"  Exclude subtypes : {exclude_subtypes}")
    print("=" * 70)

    # ── 1. Đọc 4 file ─────────────────────────────────────────────────────
    labels = pd.read_csv(os.path.join(input_dir, "final_labels.csv"),       index_col=0)
    gene   = pd.read_csv(os.path.join(input_dir, "final_gene.csv"),         index_col=0)
    mirna  = pd.read_csv(os.path.join(input_dir, "final_mirna.csv"),        index_col=0)
    methyl = pd.read_csv(os.path.join(input_dir, "final_methylation.csv"),  index_col=0)

    # Patient ID phải khớp nhau
    assert list(gene.index) == list(mirna.index) == list(methyl.index) == list(labels.index), \
        "Patient ID mismatch giữa 4 file final_*.csv ở input."

    n_original = len(labels)
    print(f"\n[Subset] Original: {n_original} samples")
    print(f"           Cancer_Type: {sorted(labels['Cancer_Type'].unique())}")
    print(f"           Subtype:     {sorted(labels['Subtype'].unique())}")

    # ── 2. Tạo mask lọc ──────────────────────────────────────────────────
    mask = pd.Series(True, index=labels.index)

    if include_cancers:
        m = labels["Cancer_Type"].isin(include_cancers)
        mask &= m
        print(f"\n[Subset] Sau khi giữ Cancer_Type {include_cancers}: {mask.sum()} samples")

    if exclude_subtypes:
        m = ~labels["Subtype"].isin(exclude_subtypes)
        mask &= m
        print(f"[Subset] Sau khi loại Subtype {exclude_subtypes}: {mask.sum()} samples")

    if mask.sum() == 0:
        raise ValueError("Filter làm rỗng dataset. Kiểm tra lại tham số.")

    # ── 3. Apply mask cho cả 4 dataframe ─────────────────────────────────
    labels_sub = labels[mask].copy()
    gene_sub   = gene[mask].copy()
    mirna_sub  = mirna[mask].copy()
    methyl_sub = methyl[mask].copy()

    # ── 4. Re-index Target_Label về 0..K-1 (contiguous) ──────────────────
    # Cần thiết vì sau khi loại class, label cũ có thể không contiguous
    # (vd: 0,1,2,4 → 0,1,2,3). MoXGATE classifier expect 0..K-1.
    old_labels_sorted = sorted(labels_sub["Target_Label"].unique())
    remap = {old: new for new, old in enumerate(old_labels_sorted)}
    labels_sub["Target_Label"] = labels_sub["Target_Label"].map(remap)

    print(f"\n[Subset] Target_Label remap (old -> new): {remap}")

    # ── 5. In phân phối sau lọc ──────────────────────────────────────────
    print(f"\n[Subset] Final: {len(labels_sub)} samples, "
          f"{labels_sub['Target_Label'].nunique()} classes")
    print(f"[Subset] Subtype distribution:")
    dist = (
        labels_sub.groupby(["Target_Label", "Subtype"])
        .size()
        .reset_index(name="count")
        .sort_values("Target_Label")
    )
    for _, row in dist.iterrows():
        print(f"           label={row['Target_Label']}  {row['Subtype']:<10} : {row['count']:>4}")

    # Cảnh báo class quá ít cho 5-fold
    min_count = labels_sub["Target_Label"].value_counts().min()
    if min_count < 5:
        print(f"\n[Subset] ⚠ Class nhỏ nhất chỉ có {min_count} sample — 5-fold CV có thể không stratify được.")
    elif min_count < 10:
        print(f"\n[Subset] ⚠ Class nhỏ nhất chỉ có {min_count} sample — fold sẽ có ít test sample/class.")

    # ── 6. Lưu output ────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    labels_sub.to_csv(os.path.join(output_dir, "final_labels.csv"))
    gene_sub.to_csv(  os.path.join(output_dir, "final_gene.csv"))
    mirna_sub.to_csv( os.path.join(output_dir, "final_mirna.csv"))
    methyl_sub.to_csv(os.path.join(output_dir, "final_methylation.csv"))

    print(f"\n[Subset] ✓ Đã lưu 4 file vào: {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Tạo subset dataset từ data_final đã có",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  type=str, required=True,
                        help="Thư mục input (chứa final_*.csv)")
    parser.add_argument("--output", type=str, required=True,
                        help="Thư mục output (sẽ tạo nếu chưa có)")
    parser.add_argument("--include_cancers", nargs="+", default=None,
                        help="Giữ lại các Cancer_Type này (vd: STAD)")
    parser.add_argument("--exclude_subtypes", nargs="+", default=None,
                        help="Loại các Subtype này (vd: HM-SNV Normal)")
    args = parser.parse_args()

    make_subset(
        input_dir=args.input,
        output_dir=args.output,
        include_cancers=args.include_cancers,
        exclude_subtypes=args.exclude_subtypes,
    )


if __name__ == "__main__":
    main()
