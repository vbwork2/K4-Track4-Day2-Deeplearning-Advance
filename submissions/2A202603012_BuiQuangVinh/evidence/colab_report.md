# DeepWeeds Lab Day 2 - 2A202603012 Bui Quang Vinh

Trạng thái: **EXPERIMENTS COMPLETED**. Các số dưới đây lấy từ artifact thật; phần thiếu giữ PENDING.

## Tóm tắt

So sánh backbone, công thức huấn luyện và inference trên fold 0. Chọn cấu hình bằng validation, khóa cấu hình rồi mới chạy test với seed 0, 1, 2.

| cấu hình | chỉ số | mean | std | seeds |
| --- | --- | --- | --- | --- |
| T00 | macro_f1 | 0.9712 | 0.0020 | 3 |
| T00 | top1 | 0.9771 | 0.0012 | 3 |
| T00 | ece | 0.0137 | 0.0008 | 3 |
| F01 | macro_f1 | 0.9798 | 0.0003 | 3 |
| F01 | top1 | 0.9832 | 0.0003 | 3 |
| F01 | ece | 0.1375 | 0.0017 | 3 |

Chênh macro-F1 test F01 - T00 = +0.0086. Đối chiếu với std giữa các seed.

## Dữ liệu và thiết lập

- DeepWeeds: 9 lớp, 17.509 ảnh. Train/val/test dùng CSV fold 0 chính thức.
- Số ảnh từng tập: `{'train': 10501, 'val': 3501, 'test': 3507}`. Giao các tập: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`.
- Sai lệch nhãn đã xác minh từ nguồn: `[{'split': 'train', 'Filename': '20170714-110407-3.jpg', 'split_label': 0, 'labels_csv_label': 1}]`. Giữ nguyên nhãn split và kiểm tra checksum.
- Ảnh đọc từ local Colab; checkpoint, log và output lưu trên Drive.
- Phiên bản thư viện, pretrained tag, epoch và batch được ghi trong `config.json` từng run.
- Biểu đồ EDA: `curves/fold0_class_counts.png`, `curves/fold0_examples.png`.

## Kiểm tra pipeline

Trạng thái: **PASS**. Kiểm tra output 9 lớp, focal gamma 0, Mixup/CutMix, optimizer groups, frozen BatchNorm, fusion và overfit batch nhỏ.

Loss batch nhỏ: `2.205204725265503` -> `0.00023684764164499938`.

## So sánh backbone

| exp_id | backbone | pretrained_tag | params_m | gmac | val_macro_f1 | best_val_top1 | mean_epoch_seconds | latency_p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B01 | resnet50 | a1_in1k | 23.5265 | 4.0872 | 0.8389 | 0.8812 | 68.8188 | 14.5710 |
| B02 | resnext50_32x4d | a1h_in1k | 22.9983 | 4.2285 | 0.8613 | 0.8940 | 75.8891 | 16.3494 |
| B03 | convnext_tiny | in12k_ft_in1k | 27.8270 | 4.4548 | 0.9665 | 0.9749 | 87.4139 | 19.8776 |
| B04 | deit_small_patch16_224 | fb_in1k | 21.6691 | 4.5985 | 0.9578 | 0.9694 | 76.7370 | 15.7061 |
| B05 | efficientnet_b0 | ra_in1k | 4.0191 | 0.3845 | 0.8864 | 0.9152 | 67.2103 | 9.5239 |

GMAC dùng Conv/Linear và tích ma trận attention của ViT; không tính các phép toán elementwise. Chọn winner bằng macro-F1 validation.

## Ablation huấn luyện

| exp_id | backbone | init | aug | loss | mix | ema_decay | val_macro_f1 | best_val_top1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T00 | convnext_tiny | finetune | basic | ce | PENDING | PENDING | 0.9665 | 0.9749 |
| T00 | convnext_tiny | finetune | basic | ce | PENDING | PENDING | 0.9654 | 0.9726 |
| T00 | convnext_tiny | finetune | basic | ce | PENDING | PENDING | 0.9727 | 0.9797 |
| T01 | convnext_tiny | frozen | basic | ce | PENDING | PENDING | 0.8521 | 0.8835 |
| T02 | convnext_tiny | scratch | basic | ce | PENDING | PENDING | 0.3059 | 0.5793 |
| T03 | convnext_tiny | finetune | color | ce | PENDING | PENDING | 0.9699 | 0.9763 |
| T04 | convnext_tiny | finetune | trivial | ce | PENDING | PENDING | 0.9676 | 0.9757 |
| T05 | convnext_tiny | finetune | basic | ls | PENDING | PENDING | 0.9722 | 0.9791 |
| T06 | convnext_tiny | finetune | basic | focal | PENDING | PENDING | 0.9644 | 0.9729 |
| T07 | convnext_tiny | finetune | basic | ce | cutmix | PENDING | 0.9708 | 0.9769 |
| T08 | convnext_tiny | finetune | basic | ce | PENDING | 0.9990 | 0.9683 | 0.9763 |
| T_COMBO | convnext_tiny | finetune | basic | ls | cutmix | 0.9990 | 0.9729 | 0.9791 |

Các lần chạy đơn yếu tố so với T00 dùng cùng backbone winner. T_COMBO kết hợp augmentation, loss và EMA được chọn từ validation. Screening một seed chỉ mang tính thăm dò.

## Inference và hiệu chuẩn

| exp_id | method | aggregation | img_size | macro_f1_val | ece_val | p95_ms | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| I00 | fp32 | prob | 224 | 0.9729 | 0.0839 | 34.8546 | RECORDED |
| I01 | hflip | prob | 224 | 0.9733 | 0.0869 | 41.1903 | RECORDED |
| I02 | multicrop | prob | 224 | 0.9647 | 0.0796 | 115.8078 | RECORDED |
| I03 | hflip | logit | 224 | 0.9733 | 0.0858 | 25.1551 | RECORDED |
| I04 | multiscale | prob | 224 | 0.9737 | 0.1046 | 29.0896 | RECORDED |
| I05 | single | prob | 224 | 0.9729 | 0.0835 | 29.8147 | RECORDED |
| I06 | fp32 | prob | 256 | 0.9750 | 0.1165 | 16.9019 | RECORDED |
| I07 | fp32 | prob | 288 | 0.9752 | 0.1361 | 13.6974 | RECORDED |
| I08 | temperature_scaling | prob | 224 | 0.9729 | 0.0058 | 12.8209 | RECORDED |

Temperature chỉ fit trên validation. TTA và latency dùng cùng hàm tạo view. Các phương pháp không chạy được được ghi FAILED cùng nguyên nhân.

## Độ trễ

| configuration | gpu | dtype | batch | img_size | p50 | p95 | p99 | images_per_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F01 locked inference: fp32 | Tesla T4 | FP32 | 1 | 288 | 31.9488 | 49.8523 | 57.3645 | 31.3000 |
| F01 locked inference: fp32 | Tesla T4 | FP32 | 32 | 288 | 232.1437 | 237.9451 | 243.3163 | 137.8457 |

Đo ít nhất 10 warmup và 50 lượt, đồng bộ GPU. Không tính đọc ảnh và resize ảnh từ đĩa; tính tạo view và forward trên tensor đã chuẩn bị.

## Kết quả cuối và phân tích lỗi

| exp_id | class | support | precision | recall | f1 | f1_std |
| --- | --- | --- | --- | --- | --- | --- |
| T00 | Chinee Apple | 226 | 0.9599 | 0.9513 | 0.9555 | 0.0081 |
| T00 | Lantana | 213 | 0.9618 | 0.9828 | 0.9722 | 0.0105 |
| T00 | Parkinsonia | 207 | 0.9823 | 0.9823 | 0.9823 | 0.0028 |
| T00 | Parthenium | 205 | 0.9886 | 0.9756 | 0.9820 | 0.0076 |
| T00 | Prickly Acacia | 213 | 0.9338 | 0.9671 | 0.9501 | 0.0041 |
| T00 | Rubber Vine | 202 | 0.9819 | 0.9802 | 0.9810 | 0.0057 |
| T00 | Siam Weed | 215 | 0.9641 | 0.9907 | 0.9771 | 0.0098 |
| T00 | Snake Weed | 204 | 0.9636 | 0.9493 | 0.9564 | 0.0081 |
| T00 | Negatives | 1822 | 0.9871 | 0.9815 | 0.9843 | 0.0008 |
| F01 | Chinee Apple | 226 | 0.9775 | 0.9587 | 0.9680 | 0.0011 |
| F01 | Lantana | 213 | 0.9936 | 0.9765 | 0.9850 | 0.0013 |
| F01 | Parkinsonia | 207 | 0.9795 | 0.9936 | 0.9864 | 0.0059 |
| F01 | Parthenium | 205 | 1.0000 | 0.9789 | 0.9893 | 0.0038 |
| F01 | Prickly Acacia | 213 | 0.9530 | 0.9828 | 0.9677 | 0.0038 |
| F01 | Rubber Vine | 202 | 0.9884 | 0.9769 | 0.9826 | 0.0001 |
| F01 | Siam Weed | 215 | 0.9891 | 0.9876 | 0.9884 | 0.0024 |
| F01 | Snake Weed | 204 | 0.9781 | 0.9493 | 0.9635 | 0.0039 |
| F01 | Negatives | 1822 | 0.9842 | 0.9903 | 0.9872 | 0.0004 |

F1 từng lớp là mean giữa các seed, không gộp thành một ensemble khác cấu hình đã khóa. Ma trận nhầm lẫn minh họa dùng seed 0. Ảnh nhầm Chinee Apple/Snake Weed nằm trong `curves/cross_class_error_examples.png` nếu có.

## Kết luận

Chênh macro-F1 test F01 - T00 = +0.0086. Đối chiếu với std giữa các seed.

p95 batch 1 = 49.852 ms trên Tesla T4; đạt mục tiêu 100 ms trong điều kiện đo.

## Hạn chế

- Chỉ dùng fold 0; split không tách theo địa điểm, nên kết quả có thể lạc quan khi triển khai nơi khác.
- Screening một seed; final ba seed. Chênh lệch nhỏ cần được đối chiếu với nhiễu giữa các seed.
- Ngân sách 10 epoch ngắn hơn nghiên cứu gốc; không suy ra rằng backbone hoặc recipe nào luôn tốt nhất.
- Latency phụ thuộc GPU và batch; cần đo lại trên phần cứng triển khai.

## Tái lập

Mở `code/lab_day2_colab.ipynb`, bật GPU và chạy từ SECTION 00. Upload `images.zip` vào `MyDrive/K4_Track4_Day2/data/images.zip`. Chi tiết cấu hình và khóa final nằm trong artifact trên Drive. Bảng đầy đủ: `results.xlsx`.
