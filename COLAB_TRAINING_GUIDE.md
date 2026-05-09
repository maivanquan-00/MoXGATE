# MoXGATE Colab Quick Run Guide

## 1) Setup (run once)
```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/ĐATN_2025.2
!pip install torch torch-geometric scikit-learn pandas numpy -q
```

## 2) Paper default hyperparameters (locked)
- epochs: 100
- batch_size: 32
- lr: 1e-4
- weight_decay: 1e-2
- lambda1: 0.01
- lambda2: 1e-4
- patience: 15
- test_ratio: 0.2
- val_ratio: 0.1
- seed: 42

These defaults are taken from config.py and used automatically by run_colab.py.

## 3) Minimal commands (copy/paste)

### GI only (Phase 1)

Note:
- gi_paper = exact paper protocol (ESCA held-out test), softmax fusion.
- gi_softmax = optional modern baseline (random 80/20 split), also softmax fusion.
- If you only want strict paper reproduction, run only gi_paper.

```python
!python MoXGATE/run_colab.py --experiment gi_paper
!python MoXGATE/run_colab.py --experiment gi_softmax
!python MoXGATE/run_colab.py --experiment gi_sparsemax
```

### All 5 datasets softmax k-fold (Phase 2.1)
```python
!python MoXGATE/run_colab.py --experiment phase2_softmax_all
```

### All 5 datasets sparsemax k-fold (Phase 2.2)
```python
!python MoXGATE/run_colab.py --experiment phase2_sparsemax_all
```

### Run everything (13 experiments)
```python
!python MoXGATE/run_colab.py --experiment all
```

## 4) Optional single-dataset k-fold

### Softmax
```python
!python MoXGATE/run_colab.py --experiment kfold_softmax --dataset brca
```

### Sparsemax
```python
!python MoXGATE/run_colab.py --experiment kfold_sparsemax --dataset lgg
```

Supported dataset values: gi, brca, ucec, kipan, lgg.

## 5) Quick test mode (low compute)

Runs only 1 fold for fast validation in k-fold scripts.

```python
!python MoXGATE/run_colab.py --experiment kfold_softmax --dataset gi --test_mode
!python MoXGATE/run_colab.py --experiment kfold_sparsemax --dataset gi --test_mode
```

## 6) Optional override (if needed)

You can override defaults without editing any training file.

```python
!python MoXGATE/run_colab.py --experiment kfold_softmax --dataset ucec --epochs 50 --batch_size 16
```

## 7) Output files
- GI paper: checkpoints_gi_paper/best_model.pt and test_results.json
- GI softmax: checkpoints_gi_softmax/best_model_gi_new.pt
- GI sparsemax: checkpoints_gi_sparsemax/best_model_sparse_gi_new.pt
- k-fold softmax: results_kfold_<dataset>_softmax.json
- k-fold sparsemax: results_kfold_<dataset>_sparsemax.json
