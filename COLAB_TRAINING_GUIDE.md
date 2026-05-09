# MoXGATE Training Guide for Colab

## Setup (Run Once at Start)
```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate and install dependencies
import os
os.chdir('/content/drive/MyDrive/ĐATN_2025.2')

# Install required packages if needed
!pip install torch torch-geometric scikit-learn pandas numpy -q
```

---

## MoXGATE Hyperparameters Explained

All parameters are **locked to paper defaults** (no changes needed). But here's what they do:

| Parameter                  | Value | Meaning                                              |
| -------------------------- | ----- | ---------------------------------------------------- |
| `--epochs`                 | 100   | Total training epochs (paper setting)                |
| `--batch_size`             | 32    | Samples per batch (paper setting)                    |
| `--lr` / `--learning_rate` | 1e-4  | Learning rate for optimizer (Adam default)           |
| `--weight_decay`           | 1e-2  | L2 regularization strength                           |
| `--lambda1`                | 0.01  | Cross-attention regularization weight                |
| `--lambda2`                | 1e-4  | Auxiliary loss weight                                |
| `--patience`               | 15    | Early stopping patience (epochs without improvement) |
| `--seed`                   | 42    | Random seed (for reproducibility)                    |
| `--test_ratio`             | 0.2   | Test set ratio (80/20 split only)                    |
| `--val_ratio`              | 0.1   | Validation set ratio (of training set)               |

**Key Points:**
- ✅ **DO NOT CHANGE** these defaults — they are from the MoXGATE paper
- All scripts already use these defaults (no need to type them)
- Optional: override with `--epochs 50` if you want quick test run

---

## Training Workflow

### **Phase 1: GI Dataset Only (Paper Reproduction)**

#### 1.1 Train with ESCA Test Set (Original Paper Setup)
**File:** `train.py`  
**Purpose:** Paper's original setup — use ESCA as test, train on COAD/READ/STAD.
```python
%cd /content/drive/MyDrive/ĐATN_2025.2
!python MoXGATE/train.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final \
    --save_dir /content/drive/MyDrive/ĐATN_2025.2/checkpoints_gi_paper \
    --epochs 100 \
    --batch_size 32 \
    --seed 42
```

#### 1.2 Train with 80/20 Split (Softmax)
**File:** `train_new.py`  
**Purpose:** Modern setup — 80/20 random split, softmax fusion.
```python
!python MoXGATE/train_new.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final \
    --save_dir /content/drive/MyDrive/ĐATN_2025.2/checkpoints_gi_softmax \
    --epochs 100 \
    --test_ratio 0.2 \
    --val_ratio 0.1 \
    --seed 42
```

#### 1.3 Train with 80/20 Split (Sparsemax)
**File:** `train_sparse_new.py`  
**Purpose:** Modern setup with sparse attention.
```python
!python MoXGATE/train_sparse_new.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final \
    --save_dir /content/drive/MyDrive/ĐATN_2025.2/checkpoints_gi_sparsemax \
    --epochs 100 \
    --test_ratio 0.2 \
    --val_ratio 0.1 \
    --seed 42
```

---

### **Phase 2: 5-Fold Cross-Validation on All 5 Datasets**

#### 2.1 Softmax K-Fold for Each Dataset
**File:** `train_kfold.py`  
**Purpose:** 5-fold CV (k=5) with softmax fusion, 80/20 split.

Run for each dataset sequentially:

```python
# GI Dataset
!python MoXGATE/train_kfold.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_gi_softmax.json \
    --epochs 100 \
    --batch_size 32 \
    --seed 42
```

```python
# BRCA Dataset
!python MoXGATE/train_kfold.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final_brca \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_brca_softmax.json \
    --epochs 100 \
    --batch_size 32 \
    --seed 42
```

```python
# UCEC Dataset
!python MoXGATE/train_kfold.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final_ucec \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_ucec_softmax.json \
    --epochs 100 \
    --batch_size 32 \
    --seed 42
```

```python
# KIPAN Dataset
!python MoXGATE/train_kfold.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final_kipan \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_kipan_softmax.json \
    --epochs 100 \
    --batch_size 32 \
    --seed 42
```

```python
# LGG Dataset
!python MoXGATE/train_kfold.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final_lgg \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_lgg_softmax.json \
    --epochs 100 \
    --batch_size 32 \
    --seed 42
```

#### 2.2 Sparsemax K-Fold for Each Dataset
**File:** `train_kfold_sparse.py`  
**Purpose:** 5-fold CV (k=5) with sparsemax fusion, 80/20 split.

Run for each dataset sequentially:

```python
# GI Dataset (Sparsemax)
!python MoXGATE/train_kfold_sparse.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_gi_sparsemax.json \
    --epochs 100 \
    --batch_size 32 \
    --seed 42
```

```python
# BRCA Dataset (Sparsemax)
!python MoXGATE/train_kfold_sparse.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final_brca \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_brca_sparsemax.json \
    --epochs 100 \
    --batch_size 32 \
    --seed 42
```

```python
# UCEC Dataset (Sparsemax)
!python MoXGATE/train_kfold_sparse.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final_ucec \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_ucec_sparsemax.json \
    --epochs 100 \
    --batch_size 32 \
    --seed 42
```

```python
# KIPAN Dataset (Sparsemax)
!python MoXGATE/train_kfold_sparse.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final_kipan \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_kipan_sparsemax.json \
    --epochs 100 \
    --batch_size 32 \
    --seed 42
```

```python
# LGG Dataset (Sparsemax)
!python MoXGATE/train_kfold_sparse.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final_lgg \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_lgg_sparsemax.json \
    --epochs 100 \
    --batch_size 32 \
    --seed 42
```

---

## Recommended Execution Order

1. **Setup** (once)
2. **Phase 1** — GI only (3 runs: paper, softmax, sparsemax)
3. **Phase 2.1** — All 5 datasets, softmax k-fold (5 runs)
4. **Phase 2.2** — All 5 datasets, sparsemax k-fold (5 runs)

---

## Complete Copy-Paste Cells (All Parameters Explicit)

### Phase 1: GI Only

**Cell 1.1 — Paper Setup (ESCA held out)**
```python
%cd /content/drive/MyDrive/ĐATN_2025.2
!python MoXGATE/train.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final \
    --save_dir /content/drive/MyDrive/ĐATN_2025.2/checkpoints_gi_paper \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --lambda1 0.01 \
    --lambda2 0.0001 \
    --seed 42
```

**Cell 1.2 — GI Softmax (80/20 split)**
```python
!python MoXGATE/train_new.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final \
    --save_dir /content/drive/MyDrive/ĐATN_2025.2/checkpoints_gi_softmax \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --lambda1 0.01 \
    --lambda2 0.0001 \
    --test_ratio 0.2 \
    --val_ratio 0.1 \
    --seed 42
```

**Cell 1.3 — GI Sparsemax (80/20 split)**
```python
!python MoXGATE/train_sparse_new.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final \
    --save_dir /content/drive/MyDrive/ĐATN_2025.2/checkpoints_gi_sparsemax \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --lambda1 0.01 \
    --lambda2 0.0001 \
    --test_ratio 0.2 \
    --val_ratio 0.1 \
    --seed 42
```

### Phase 2.1: All 5 Datasets — Softmax K-Fold

**Cell 2.1a — GI (softmax, 5-fold)**
```python
!python MoXGATE/train_kfold.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_gi_softmax.json \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --lambda1 0.01 \
    --lambda2 0.0001 \
    --seed 42
```

**Cell 2.1b — BRCA (softmax, 5-fold)**
```python
!python MoXGATE/train_kfold.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final_brca \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_brca_softmax.json \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --lambda1 0.01 \
    --lambda2 0.0001 \
    --seed 42
```

**Cell 2.1c — UCEC (softmax, 5-fold)**
```python
!python MoXGATE/train_kfold.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final_ucec \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_ucec_softmax.json \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --lambda1 0.01 \
    --lambda2 0.0001 \
    --seed 42
```

**Cell 2.1d — KIPAN (softmax, 5-fold)**
```python
!python MoXGATE/train_kfold.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final_kipan \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_kipan_softmax.json \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --lambda1 0.01 \
    --lambda2 0.0001 \
    --seed 42
```

**Cell 2.1e — LGG (softmax, 5-fold)**
```python
!python MoXGATE/train_kfold.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final_lgg \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_lgg_softmax.json \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --lambda1 0.01 \
    --lambda2 0.0001 \
    --seed 42
```

### Phase 2.2: All 5 Datasets — Sparsemax K-Fold

**Cell 2.2a — GI (sparsemax, 5-fold)**
```python
!python MoXGATE/train_kfold_sparse.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_gi_sparsemax.json \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --lambda1 0.01 \
    --lambda2 0.0001 \
    --seed 42
```

**Cell 2.2b — BRCA (sparsemax, 5-fold)**
```python
!python MoXGATE/train_kfold_sparse.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final_brca \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_brca_sparsemax.json \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --lambda1 0.01 \
    --lambda2 0.0001 \
    --seed 42
```

**Cell 2.2c — UCEC (sparsemax, 5-fold)**
```python
!python MoXGATE/train_kfold_sparse.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final_ucec \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_ucec_sparsemax.json \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --lambda1 0.01 \
    --lambda2 0.0001 \
    --seed 42
```

**Cell 2.2d — KIPAN (sparsemax, 5-fold)**
```python
!python MoXGATE/train_kfold_sparse.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final_kipan \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_kipan_sparsemax.json \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --lambda1 0.01 \
    --lambda2 0.0001 \
    --seed 42
```

**Cell 2.2e — LGG (sparsemax, 5-fold)**
```python
!python MoXGATE/train_kfold_sparse.py \
    --data_dir /content/drive/MyDrive/ĐATN_2025.2/data_final_lgg \
    --save_path /content/drive/MyDrive/ĐATN_2025.2/results_kfold_lgg_sparsemax.json \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --lambda1 0.01 \
    --lambda2 0.0001 \
    --seed 42
```

---

## Results Location

| Experiment               | Output                                                     |
| ------------------------ | ---------------------------------------------------------- |
| GI Paper (ESCA test)     | `checkpoints_gi_paper/best_model.pt` + `test_results.json` |
| GI Softmax 80/20         | `checkpoints_gi_softmax/best_model_gi_new.pt`              |
| GI Sparsemax 80/20       | `checkpoints_gi_sparsemax/best_model_sparse_gi_new.pt`     |
| K-Fold Softmax (All 5)   | `results_kfold_*_softmax.json`                             |
| K-Fold Sparsemax (All 5) | `results_kfold_*_sparsemax.json`                           |

---

## Summary

```
Training Scripts (5 total):
  1. train.py              → GI only, ESCA held out (paper reproduction)
  2. train_new.py          → GI only, 80/20 split (softmax)
  3. train_sparse_new.py   → GI only, 80/20 split (sparsemax)
  4. train_kfold.py        → All 5 datasets, 5-fold CV (softmax)
  5. train_kfold_sparse.py → All 5 datasets, 5-fold CV (sparsemax) [NEW]
```

**Total Experiments: 13**
- Phase 1 (GI): 3 experiments
- Phase 2.1 (All 5 datasets, softmax): 5 experiments
- Phase 2.2 (All 5 datasets, sparsemax): 5 experiments
