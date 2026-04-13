"""
config.py
=========
Cấu hình đường dẫn trung tâm cho toàn bộ dự án MoXGATE.

╔══════════════════════════════════════════════════════════════════╗
║  ĐỌC TRƯỚC KHI DÙNG                                              ║
║  File này TỰ ĐỘNG phát hiện môi trường (Local / Google Colab)    ║
║  và thiết lập đường dẫn tương ứng.                               ║
║                                                                  ║
║  Nếu bạn chỉ muốn thay đổi đường dẫn:                            ║
║    → Chỉnh sửa phần "CẤU HÌNH THỦ CÔNG" bên dưới                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys


# ══════════════════════════════════════════════════════════════════
#  CẤU HÌNH THỦ CÔNG — Chỉnh sửa tại đây nếu cần
# ══════════════════════════════════════════════════════════════════

# Đường dẫn trên máy LOCAL (Windows / Linux / Mac)
LOCAL_BASE = r"d:\ĐATN\MoXGATE"

# Tên thư mục trên Google Drive (Colab)
# Nếu Drive của bạn có cấu trúc: /content/drive/MyDrive/<DRIVE_PROJECT_DIR>/
DRIVE_PROJECT_DIR = "ĐATN_2025.2"

# ══════════════════════════════════════════════════════════════════
#  TỰ ĐỘNG PHÁT HIỆN MÔI TRƯỜNG — Không cần chỉnh sửa bên dưới
# ══════════════════════════════════════════════════════════════════

def _is_colab() -> bool:
    """Kiểm tra có đang chạy trên Google Colab không."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_base_dir() -> str:
    """
    Tự động xác định BASE_DIR:
      - Colab: /content/drive/MyDrive/<DRIVE_PROJECT_DIR>/
      - Local: LOCAL_BASE
    """
    if _is_colab():
        drive_root = "/content/drive/MyDrive"
        base = os.path.join(drive_root, DRIVE_PROJECT_DIR)
        # Nếu Drive chưa được mount, cảnh báo
        if not os.path.isdir(drive_root):
            print(
                "[config] ⚠️  Google Drive chưa được mount!\n"
                "         Chạy lệnh sau trong Colab trước khi dùng:\n"
                "         from google.colab import drive; drive.mount('/content/drive')"
            )
        return base
    else:
        return os.path.abspath(LOCAL_BASE)


# ── Base directory ────────────────────────────────────────────────
BASE_DIR = _detect_base_dir()
IS_COLAB = _is_colab()

# ── Thư mục CODE (nơi chứa các file .py của repo) ─────────────────
# Colab: file config.py nằm cùng thư mục với train.py, dataset.py...
#   → lấy thư mục chứa chính file config.py này làm CODE_DIR
# Local: giống BASE_DIR
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Thêm CODE_DIR vào sys.path để import luôn hoạt động ───────────
# Cho phép chạy từ bất kỳ thư mục nào:
#   %cd /content        → !python MoXGATE/train.py   ✓
#   %cd /content/MoXGATE → !python train.py           ✓
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


# ══════════════════════════════════════════════════════════════════
#  ĐƯỜNG DẪN DỮ LIỆU — GI (Gastrointestinal Cancer)
# ══════════════════════════════════════════════════════════════════

# Dữ liệu gốc — đọc vào, không ghi
GI_RAW_OMICS_DIR  = os.path.join(BASE_DIR, "data_original", "multi_omics")
GI_RAW_SUBTYPE_DIR= os.path.join(BASE_DIR, "data_original", "subtype")
ANNOTATION_DIR    = os.path.join(BASE_DIR, "data_original", "annotation")
GTF_PATH          = os.path.join(ANNOTATION_DIR, "gencode.v36.annotation.gtf")
CROSS_REACTIVE_PATH = os.path.join(ANNOTATION_DIR, "cross_reactive_probes.txt")
MANIFEST_PATH     = os.path.join(ANNOTATION_DIR, "HumanMethylation450_manifest.csv")

# Dữ liệu trung gian (sau preprocessing)
GI_PROCESSED_DIR  = os.path.join(BASE_DIR, "data_processed")
GI_LABELS_PATH    = os.path.join(GI_PROCESSED_DIR, "clean_labels.csv")

# Dữ liệu cuối (sẵn sàng để train)
GI_FINAL_DIR      = os.path.join(BASE_DIR, "data_final")

# Checkpoint & kết quả
GI_CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")


# ══════════════════════════════════════════════════════════════════
#  ĐƯỜNG DẪN DỮ LIỆU — BRCA (Breast Cancer)
# ══════════════════════════════════════════════════════════════════

# Dữ liệu gốc
BRCA_RAW_OMICS_DIR  = os.path.join(BASE_DIR, "data_original", "multi_omics_brca")
BRCA_RAW_SUBTYPE_DIR= os.path.join(BASE_DIR, "data_original", "subtype_brca")

# Dữ liệu trung gian
BRCA_PROCESSED_DIR  = os.path.join(BASE_DIR, "data_processed_brca")
BRCA_LABELS_PATH    = os.path.join(BRCA_PROCESSED_DIR, "clean_labels_brca.csv")

# Dữ liệu cuối
BRCA_FINAL_DIR      = os.path.join(BASE_DIR, "data_final_brca")

# Checkpoint & kết quả
BRCA_CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_brca")


# ══════════════════════════════════════════════════════════════════
#  ĐƯỜNG DẪN DỮ LIỆU — KIPAN (Kidney Cancer: KICH / KIRC / KIRP)
# ══════════════════════════════════════════════════════════════════

# Dữ liệu gốc
KIPAN_RAW_OMICS_DIR   = os.path.join(BASE_DIR, "data_original", "multi_omics_kipan")
KIPAN_RAW_SUBTYPE_DIR = os.path.join(BASE_DIR, "data_original", "subtype_kipan")

# Dữ liệu trung gian
KIPAN_PROCESSED_DIR   = os.path.join(BASE_DIR, "data_processed_kipan")
KIPAN_LABELS_PATH     = os.path.join(KIPAN_PROCESSED_DIR, "clean_labels_kipan.csv")

# Dữ liệu cuối
KIPAN_FINAL_DIR       = os.path.join(BASE_DIR, "data_final_kipan")

# Checkpoint & kết quả
KIPAN_CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints_kipan")


# ══════════════════════════════════════════════════════════════════
#  ĐƯỜNG DẪN DỮ LIỆU — UCEC (Uterine Endometrial Carcinoma)
# ══════════════════════════════════════════════════════════════════

# Dữ liệu gốc
UCEC_RAW_OMICS_DIR   = os.path.join(BASE_DIR, "data_original", "multi_omics_ucec")
UCEC_RAW_SUBTYPE_DIR = os.path.join(BASE_DIR, "data_original", "subtype_ucec")

# Dữ liệu trung gian
UCEC_PROCESSED_DIR   = os.path.join(BASE_DIR, "data_processed_ucec")
UCEC_LABELS_PATH     = os.path.join(UCEC_PROCESSED_DIR, "clean_labels_ucec.csv")

# Dữ liệu cuối
UCEC_FINAL_DIR       = os.path.join(BASE_DIR, "data_final_ucec")

# Checkpoint & kết quả
UCEC_CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints_ucec")


# ══════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS MẶC ĐỊNH (theo paper MoXGATE)
# ══════════════════════════════════════════════════════════════════

DEFAULT_BATCH_SIZE   = 32
DEFAULT_EPOCHS       = 100
DEFAULT_LR           = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_LAMBDA1      = 0.01
DEFAULT_LAMBDA2      = 1e-4
DEFAULT_PATIENCE     = 15
DEFAULT_VAL_RATIO    = 0.08
DEFAULT_TEST_RATIO   = 0.2
DEFAULT_SEED         = 42
DEFAULT_NUM_WORKERS  = 0


# ══════════════════════════════════════════════════════════════════
#  TIỆN ÍCH
# ══════════════════════════════════════════════════════════════════

def print_config():
    """In toàn bộ cấu hình đường dẫn ra màn hình để kiểm tra."""
    import sys
    # Đảm bảo stdout dùng UTF-8 trên Windows (tránh lỗi cp1252 với ký tự tiếng Việt)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    env  = "Google Colab" if IS_COLAB else "Local"
    sep  = "=" * 60
    dash = "-" * 60
    lines = [
        f"\n{sep}",
        f"  MoXGATE Config  [{env}]",
        sep,
        f"  BASE_DIR        : {BASE_DIR}",
        f"  CODE_DIR        : {CODE_DIR}",
        dash,
        "  [GI]",
        f"  Raw omics       : {GI_RAW_OMICS_DIR}",
        f"  Processed       : {GI_PROCESSED_DIR}",
        f"  Final           : {GI_FINAL_DIR}",
        f"  Checkpoints     : {GI_CHECKPOINT_DIR}",
        dash,
        "  [BRCA]",
        f"  Raw omics       : {BRCA_RAW_OMICS_DIR}",
        f"  Processed       : {BRCA_PROCESSED_DIR}",
        f"  Final           : {BRCA_FINAL_DIR}",
        f"  Checkpoints     : {BRCA_CHECKPOINT_DIR}",
        dash,
        "  [KIPAN]",
        f"  Raw omics       : {KIPAN_RAW_OMICS_DIR}",
        f"  Processed       : {KIPAN_PROCESSED_DIR}",
        f"  Final           : {KIPAN_FINAL_DIR}",
        f"  Checkpoints     : {KIPAN_CHECKPOINT_DIR}",
        dash,
        "  [UCEC]",
        f"  Raw omics       : {UCEC_RAW_OMICS_DIR}",
        f"  Processed       : {UCEC_PROCESSED_DIR}",
        f"  Final           : {UCEC_FINAL_DIR}",
        f"  Checkpoints     : {UCEC_CHECKPOINT_DIR}",
        f"{sep}\n",
    ]
    output = "\n".join(lines)
    try:
        print(output)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((output + "\n").encode("utf-8"))


def ensure_dirs(*dirs):
    """Tạo các thư mục output nếu chưa tồn tại."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    print_config()
