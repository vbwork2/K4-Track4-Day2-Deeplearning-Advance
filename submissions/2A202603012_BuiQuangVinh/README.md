# DeepWeeds Lab Day 2 - 2A202603012 Bui Quang Vinh

**Đã chạy xong 20 lượt huấn luyện / 200 epoch:** 5 backbone, 10 recipe seed 0, 9 cấu hình inference. Baseline và final đều đủ 3 seed (0, 1, 2); benchmark final batch 1/32 trên Tesla T4 đã có kết quả.

Kết quả ngày 05/10/2026: macro-F1 test baseline **97,12 ± 0,20%**, final **97,98 ± 0,03%**; tăng **0,86 điểm phần trăm**. Top-1 final **98,32 ± 0,03%**. Std là std mẫu giữa 3 seed.

p95 final batch 1 **49,85 ms** trên Tesla T4, chưa gồm đọc/resize ảnh; throughput batch 32 **137,85 ảnh/giây**. ECE test final **13,75 ± 0,17%**, nên xác suất vẫn cần hiệu chuẩn nếu triển khai.

Chi tiết trong [report.md](report.md) và [results.xlsx](results.xlsx). Tự chấm mục I: **18/19 điểm ở các ý đã chấm** (tối đa 20); I4a chưa chấm vì final không dùng temperature scaling. Đây là điểm đề xuất từ evaluator, cần giảng viên xác nhận.

## Chạy Colab

1. Mở notebook: https://drive.google.com/file/d/1C0bj1mXGq-Bjlmkpze5kO5uGylrssXXd/view
2. Chọn runtime GPU.
3. Đặt `images.zip` tại `MyDrive/K4_Track4_Day2/data/images.zip`.
4. Chạy từ **SECTION 00**, hoặc chọn Run All.

Notebook có 44 cell. Markdown dùng tiếng Việt ngắn; code được tách dòng và comment bằng tiếng Anh.

## Các phase

| Section | Công việc |
|---|---|
| 00-04 | GPU, Drive, code, CSV chính thức và ảnh local |
| 05-06 | EDA và kiểm tra pipeline trước khi train |
| 07-09 | Recipe nền, năm backbone và chọn winner bằng val |
| 10-11 | Ablation đơn yếu tố và tổ hợp yếu tố thắng |
| 12-13 | Inference trên val và FINAL CONFIG LOCK |
| 14-17 | Final/baseline ba seed, test, latency và phân tích lỗi |
| 18-19 | Excel, báo cáo tiếng Việt và ZIP bài nộp |

Recipe chung: batch 32, 10 epoch, AMP, AdamW, warmup và cosine. Screening seed 0; final và baseline seed 0, 1, 2. T00 dùng backbone winner và recipe nền. Mọi lựa chọn dùng validation.

## Khi phiên bị ngắt

Mở lại notebook và chạy từ SECTION 00. Ảnh local đầy đủ được giữ; ảnh thiếu được khôi phục. Notebook lấy lại artifact từ Drive, kiểm tra cấu hình, rồi resume `last.pt` hoặc skip run đủ artifact.

Giữ `FORCE_RERUN=False`. Nếu đổi recipe của một experiment đã chạy, dùng exp_id mới. Sau khi có test output, không được chạy lại quá trình chọn cấu hình. Test CSV đã tồn tại được kiểm tra và bỏ qua.

## Dữ liệu chính thức

Fold 0 dùng `labels.csv`, `train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv` từ AlexOlsen/DeepWeeds. Kiểm tra SHA-256 cho CSV và MD5 cho ZIP.

Nguồn có một ngoại lệ: `20170714-110407-3.jpg` có train label 0 nhưng `labels.csv` label 1. Code chỉ chấp nhận ngoại lệ này khi checksum nguồn khớp, giữ nguyên nhãn train và ghi vào `fold0_split_check.json`.

## Output trên Drive

Workspace: `MyDrive/K4_Track4_Day2/`.

- `source/`: code và notebook hiện hành.
- `runs/<exp_id>/seed<seed>/`: config, history, checkpoint, logit và summary.
- `predictions/`, `curves/`, `eval_out/`: prediction và biểu đồ thật.
- `results/`: khóa final, sanity checks, inference, latency, Excel và báo cáo.
- `submission/2A202603012_BuiQuangVinh/`: bài nộp; ZIP nằm bên cạnh.

Checkpoint và history được sync sau mỗi epoch. Gói bài nộp không chứa dataset hay checkpoint lớn. `eval.py` giữ nguyên từ repository.

## Thư viện và kiểm tra

Giữ torch/torchvision của Colab để bảo toàn CUDA. Các thư viện còn lại nằm trong `code/requirements-colab.txt`; phiên bản thực tế và pretrained tag lưu trong config từng run.

Chạy kiểm tra runtime nhỏ:

```bash
python code/runtime_checks.py
```

Chạy public tests từ gốc repository:

```bash
python -X utf8 -m unittest discover -s tests -v
```

Các test dùng dữ liệu nhỏ và thư mục tạm. Kết quả test code không được dùng làm metric cho bài lab. Bảng kết quả và report hiện lấy từ artifact Colab thật, đã đối chiếu toàn bộ 26 CSV dự đoán với nhãn fold 0 chính thức.

## Bằng chứng đã đưa vào bài nộp

- `evidence/<exp_id>/seed<k>/`: config, history, summary của 20 lượt hoàn thành.
- `evidence/evaluation/`: chỉ số tính lại bằng `eval.py` nguyên bản, F1 từng lớp và ma trận nhầm lẫn.
- `evidence/labels/`: CSV fold 0 tham chiếu, giữ nguyên byte và kiểm tra SHA-256.
- `predictions/`: 20 CSV validation và 6 CSV test, đủ baseline/final 3 seed.
- `curves/`: biểu đồ training, EDA và ảnh lỗi lấy từ dataset gốc.
- `evidence/artifact_sources.json`: ID Drive, đường dẫn và SHA-256 để truy ngược artifact.
- `evidence/latency.json`: benchmark final trên Tesla T4, batch 1/32, 10 warmup và 50 lượt đo.
- `evidence/colab_eval_out/`: output đánh giá cuối của Colab, tự chấm mục I và danh sách lỗi của cả ba seed.
- `evidence/colab_report.md`: bản báo cáo tự sinh từ lần chạy Colab hoàn tất.

Trong thư mục bài nộp, có thể tính lại kết quả mà không cần GPU:

```bash
python -X utf8 code/eval.py score --pred "predictions/F01_seed*_test.csv" --test-csv evidence/labels/test_subset0.csv --labels evidence/labels/labels.csv --tag F01
```

Lệnh tính lại đủ 3 seed mà không train hoặc dùng GPU. Giữ `FORCE_RERUN=False` khi mở lại Colab để bỏ qua các lượt đã hoàn thành. Không cần chạy training lại để đọc báo cáo.
