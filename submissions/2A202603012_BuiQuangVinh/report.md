# DeepWeeds Lab Day 2 - 2A202603012 Bui Quang Vinh

Trạng thái: **EXPERIMENTS PENDING / IN PROGRESS**. Các số dưới đây lấy từ artifact thật; phần thiếu giữ PENDING.

## Tóm tắt

So sánh backbone, công thức huấn luyện và inference trên fold 0. Chọn cấu hình bằng validation, khóa cấu hình rồi mới chạy test với seed 0, 1, 2.

PENDING: chưa có kết quả.

PENDING: cần đủ 3 seed cho baseline và final.

## Dữ liệu và thiết lập

- DeepWeeds: 9 lớp, 17.509 ảnh. Train/val/test dùng CSV fold 0 chính thức.
- Số ảnh từng tập: `PENDING`. Giao các tập: `PENDING`.
- Sai lệch nhãn đã xác minh từ nguồn: `PENDING`. Giữ nguyên nhãn split và kiểm tra checksum.
- Ảnh đọc từ local Colab; checkpoint, log và output lưu trên Drive.
- Phiên bản thư viện, pretrained tag, epoch và batch được ghi trong `config.json` từng run.
- Biểu đồ EDA: `curves/fold0_class_counts.png`, `curves/fold0_examples.png`.

## Kiểm tra pipeline

Trạng thái: **PENDING**. Kiểm tra output 9 lớp, focal gamma 0, Mixup/CutMix, optimizer groups, frozen BatchNorm, fusion và overfit batch nhỏ.

Loss batch nhỏ: `PENDING` -> `PENDING`.

## So sánh backbone

PENDING: chưa có kết quả.

GMAC dùng Conv/Linear và tích ma trận attention của ViT; không tính các phép toán elementwise. Chọn winner bằng macro-F1 validation.

## Ablation huấn luyện

PENDING: chưa có kết quả.

Các lần chạy đơn yếu tố so với T00 dùng cùng backbone winner. T_COMBO kết hợp augmentation, loss và EMA được chọn từ validation. Screening một seed chỉ mang tính thăm dò.

## Inference và hiệu chuẩn

PENDING: chưa có kết quả.

Temperature chỉ fit trên validation. TTA và latency dùng cùng hàm tạo view. Các phương pháp không chạy được được ghi FAILED cùng nguyên nhân.

## Độ trễ

PENDING: chưa có kết quả.

Đo ít nhất 10 warmup và 50 lượt, đồng bộ GPU. Không tính đọc ảnh và resize ảnh từ đĩa; tính tạo view và forward trên tensor đã chuẩn bị.

## Kết quả cuối và phân tích lỗi

PENDING: chưa có kết quả.

F1 từng lớp là mean giữa các seed, không gộp thành một ensemble khác cấu hình đã khóa. Ma trận nhầm lẫn minh họa dùng seed 0. Ảnh nhầm Chinee Apple/Snake Weed nằm trong `curves/cross_class_error_examples.png` nếu có.

## Kết luận

PENDING: cần đủ 3 seed cho baseline và final.

PENDING: cần latency và kết quả cuối.

## Hạn chế

- Chỉ dùng fold 0; split không tách theo địa điểm, nên kết quả có thể lạc quan khi triển khai nơi khác.
- Screening một seed; final ba seed. Chênh lệch nhỏ cần được đối chiếu với nhiễu giữa các seed.
- Ngân sách 10 epoch ngắn hơn nghiên cứu gốc; không suy ra rằng backbone hoặc recipe nào luôn tốt nhất.
- Latency phụ thuộc GPU và batch; cần đo lại trên phần cứng triển khai.

## Tái lập

Mở `code/lab_day2_colab.ipynb`, bật GPU và chạy từ SECTION 00. Upload `images.zip` vào `MyDrive/K4_Track4_Day2/data/images.zip`. Chi tiết cấu hình và khóa final nằm trong artifact trên Drive. Bảng đầy đủ: `results.xlsx`.
