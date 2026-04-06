# Cancer Subtype Classification: Multi-Omics Translation & Cross-Attention Integration

Dự án này triển khai một khung nghiên cứu tích hợp dữ liệu đa nền tảng (Multi-omics) chia làm hai giai đoạn kế thừa từ hai nghiên cứu tiên tiến: **Subtype-MGTP** (2024) và **MoXGATE** (2025).

---

## Tổng quan kiến trúc

Mô hình được thiết kế nhằm giải quyết bài toán thiếu hụt dữ liệu Proteomics và tối ưu hóa việc phân loại ung thư:

1.  **Phase A (Translation):** Sử dụng mạng GCN để dịch mã từ 4 nguồn Genomics sang 455 đặc trưng Protein.
2.  **Phase B (Classification):** Tích hợp 5 nguồn dữ liệu (bao gồm Protein dự đoán) thông qua cơ chế **Modality-Aware Cross-Attention** để phân loại.

---

## Cấu trúc thư mục chính

* `preprocess.py`: Script xử lý thô dữ liệu từ TCGA.
* `models.py` & `utils.py`: Kiến trúc GCN và xử lý đồ thị cho Phase A.
* `A_train.py` & `A_main.py`: Huấn luyện và trích xuất đặc trưng Protein.
* `B_models.py`: Kiến trúc MoXGATE (Self-Attention & Cross-Attention).
* `B_train_main.py`: Script huấn luyện phân loại chính cho Phase B.
* `processed_data_common/`: Thư mục chứa dữ liệu đã đồng bộ hóa.

---

## Hướng dẫn thực hiện

### 1. Tiền xử lý dữ liệu
Trước tiên, cần làm sạch và đồng bộ hóa các mẫu bệnh nhân giữa các nền tảng RNA, miRNA, Meth và CNV.

```bash
python preprocess.py
```

### 2. Chạy Phase A (Dự đoán Proteomics)
Giai đoạn này xây dựng đồ thị tương quan bệnh nhân bằng khoảng cách Cosine và huấn luyện mạng GCN để cực tiểu hóa sai số MSE giữa protein dự đoán và protein thật.

```bash
python A_train.py
```
File này sẽ đồng thời thực hiện việc huấn luyện mô hình dự đoán protein (từ processed_data_common) và thực hiện dự đoán dữ liệu protein cho bệnh nhân còn thiếu, kết quả sẽ nằm ở folder processed_data_common_no_protein.

### 3. Chạy Phase B 
Sử dụng 5 nguồn dữ liệu để thực hiện các task phân loại như PAM50, Stage, N-stage. Mô hình sử dụng 32 heads Cross-Attention và Focal Loss để xử lý mất cân bằng dữ liệu.
* Huấn luyện phân loại:

```bash
python B_train_main.py
```

### Tài liệu tham khảo
* [1] Xie, M., et al. (2024). Subtype-MGTP: a cancer subtype identification framework based on multi-omics translation. Genome Analysis.

* [2] Dip, S. A., et al. (2025). MoXGATE: Modality-Aware Cross-Attention for Multi-Omic Gastrointestinal Cancer Sub-type Classification. AI4NA workshop at ICLR 2025.