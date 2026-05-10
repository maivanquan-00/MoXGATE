"""
run_colab.py
============
Launcher script to run MoXGATE experiments with short commands on Colab/local.

Examples:
  python run_colab.py --experiment gi_paper
  python run_colab.py --experiment gi_softmax
  python run_colab.py --experiment gi_sparsemax
  python run_colab.py --experiment kfold_softmax --dataset brca
  python run_colab.py --experiment kfold_sparsemax --dataset lgg
  python run_colab.py --experiment phase1_all
  python run_colab.py --experiment phase2_softmax_all
  python run_colab.py --experiment phase2_sparsemax_all
  python run_colab.py --experiment all
"""

import argparse
import os
import subprocess
import sys
from typing import Dict

import config


DATASET_DIRS: Dict[str, str] = {
    "gi": config.GI_FINAL_DIR,
    "brca": config.BRCA_FINAL_DIR,
    "ucec": config.UCEC_FINAL_DIR,
    "kipan": config.KIPAN_FINAL_DIR,
    "lgg": config.LGG_FINAL_DIR,
}


def _run(cmd):
    print("\n" + "=" * 80)
    print("RUN:", " ".join(cmd))
    print("=" * 80)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def run_gi_paper(args):
    save_dir = os.path.join(config.BASE_DIR, "checkpoints_gi_paper")
    cmd = [
        sys.executable,
        os.path.join(config.CODE_DIR, "train.py"),
        "--data_dir", DATASET_DIRS["gi"],
        "--save_dir", save_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--lambda1", str(args.lambda1),
        "--lambda2", str(args.lambda2),
        "--seed", str(args.seed),
    ]
    _run(cmd)


def run_gi_softmax(args):
    save_dir = os.path.join(config.BASE_DIR, "checkpoints_gi_softmax")
    cmd = [
        sys.executable,
        os.path.join(config.CODE_DIR, "train_new.py"),
        "--data_dir", DATASET_DIRS["gi"],
        "--save_dir", save_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--lambda1", str(args.lambda1),
        "--lambda2", str(args.lambda2),
        "--test_ratio", str(args.test_ratio),
        "--val_ratio", str(args.val_ratio),
        "--seed", str(args.seed),
    ]
    _run(cmd)


def run_gi_sparsemax(args):
    save_dir = os.path.join(config.BASE_DIR, "checkpoints_gi_sparsemax")
    cmd = [
        sys.executable,
        os.path.join(config.CODE_DIR, "train_sparse_new.py"),
        "--data_dir", DATASET_DIRS["gi"],
        "--save_dir", save_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--lambda1", str(args.lambda1),
        "--lambda2", str(args.lambda2),
        "--test_ratio", str(args.test_ratio),
        "--val_ratio", str(args.val_ratio),
        "--seed", str(args.seed),
    ]
    _run(cmd)


def run_kfold_softmax(args, dataset: str):
    data_dir = DATASET_DIRS[dataset]
    save_path = os.path.join(config.BASE_DIR, f"results_kfold_{dataset}_softmax.json")
    cmd = [
        sys.executable,
        os.path.join(config.CODE_DIR, "train_kfold.py"),
        "--data_dir", data_dir,
        "--save_path", save_path,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--lambda1", str(args.lambda1),
        "--lambda2", str(args.lambda2),
        "--seed", str(args.seed),
    ]
    if args.multi_seed:
        cmd.append("--multi_seed")
    if args.test_mode:
        cmd.append("--test_mode")
    _run(cmd)


def run_kfold_sparsemax(args, dataset: str):
    data_dir = DATASET_DIRS[dataset]
    save_path = os.path.join(config.BASE_DIR, f"results_kfold_{dataset}_sparsemax.json")
    cmd = [
        sys.executable,
        os.path.join(config.CODE_DIR, "train_kfold_sparse.py"),
        "--data_dir", data_dir,
        "--save_path", save_path,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--lambda1", str(args.lambda1),
        "--lambda2", str(args.lambda2),
        "--seed", str(args.seed),
    ]
    if args.multi_seed:
        cmd.append("--multi_seed")
    if args.test_mode:
        cmd.append("--test_mode")
    _run(cmd)


def parse_args():
    parser = argparse.ArgumentParser(description="MoXGATE experiment launcher")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            "gi_paper",
            "gi_softmax",
            "gi_sparsemax",
            "kfold_softmax",
            "kfold_sparsemax",
            "phase1_all",
            "phase2_softmax_all",
            "phase2_sparsemax_all",
            "all",
        ],
        help="Experiment preset to run",
    )
    parser.add_argument(
        "--dataset",
        default="gi",
        choices=["gi", "brca", "ucec", "kipan", "lgg"],
        help="Used with kfold_softmax/kfold_sparsemax",
    )

    # Keep paper defaults as central defaults.
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=config.DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--lambda1", type=float, default=config.DEFAULT_LAMBDA1)
    parser.add_argument("--lambda2", type=float, default=config.DEFAULT_LAMBDA2)
    parser.add_argument("--test_ratio", type=float, default=config.DEFAULT_TEST_RATIO)
    parser.add_argument(
        "--val_ratio", type=float, default=0.08,
        help="Validation fraction of the full dataset for stratified 80/20 runs; gi_paper uses train.py's 0.1 default.",
    )
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--multi_seed", action="store_true", help="Run legacy 3-seed mode for k-fold (15 runs)")
    parser.add_argument("--test_mode", action="store_true", help="Run quick kfold test mode")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = ["gi", "brca", "ucec", "kipan", "lgg"]

    if args.experiment == "gi_paper":
        run_gi_paper(args)
    elif args.experiment == "gi_softmax":
        run_gi_softmax(args)
    elif args.experiment == "gi_sparsemax":
        run_gi_sparsemax(args)
    elif args.experiment == "kfold_softmax":
        run_kfold_softmax(args, args.dataset)
    elif args.experiment == "kfold_sparsemax":
        run_kfold_sparsemax(args, args.dataset)
    elif args.experiment == "phase1_all":
        run_gi_paper(args)
        run_gi_softmax(args)
        run_gi_sparsemax(args)
    elif args.experiment == "phase2_softmax_all":
        for ds in datasets:
            run_kfold_softmax(args, ds)
    elif args.experiment == "phase2_sparsemax_all":
        for ds in datasets:
            run_kfold_sparsemax(args, ds)
    elif args.experiment == "all":
        run_gi_paper(args)
        run_gi_softmax(args)
        run_gi_sparsemax(args)
        for ds in datasets:
            run_kfold_softmax(args, ds)
        for ds in datasets:
            run_kfold_sparsemax(args, ds)

    print("\nDone.")


if __name__ == "__main__":
    main()
