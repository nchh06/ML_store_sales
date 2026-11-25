# Dự báo doanh số Favorita

File nộp là `submission.csv`. Báo cáo thuyết trình là `report.pptx`. Cả hai được tạo từ lần chạy local trên dữ liệu Favorita. Đề bài nằm trong `docs/Final_INT3405E_5.pdf`.

## Nộp trên Kaggle

1. Đăng nhập và tham gia [Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting). Theo đề, trưởng nhóm đặt tên đội theo mẫu `UET_[INT3405E_5]_[số thứ tự nhóm]`.
2. Trong mục **Submit Predictions**, chọn `submission.csv`. Nếu nộp từ notebook, import `store_sales.ipynb`, gắn dữ liệu cuộc thi, chọn **Save Version → Save & Run All**. Notebook xuất `/kaggle/working/submission.csv`.
3. Ghi nhận RMSLE do Kaggle trả về và cập nhật slide kết quả trước khi thuyết trình.

## Tái lập trên máy cá nhân

Dùng Python 3.12. Đặt `train.csv`, `test.csv` và `sample_submission.csv` vào `data/`. Có thể giải nén ba file từ [trang dữ liệu Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data). Lần chạy local dùng [bản sao dữ liệu của t4tiana, revision 5768b1d](https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting/tree/5768b1d).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python forecast.py --data data --out .
python -m unittest -v test_forecast
python make_report.py
```

`forecast.py` xuất `results.json` và thay thế `submission.csv`. Dùng `--out` để chọn thư mục kết quả khác. `make_report.py` đọc `results.json` trong thư mục hiện tại và tạo lại báo cáo. Notebook chứa mã dự báo độc lập, dùng thư viện có sẵn của Kaggle. Để tái lập theo byte, dùng môi trường local đã ghim phiên bản. SHA-256 của file nộp đã kiểm tra là `bdb0c7f90454d2347a8f58b3fbda82c949eb8b2a51a92a658f7090504750f3c9`.

## Kết quả kiểm định

Mỗi mốc huấn luyện dùng 10 cửa sổ đích gần nhất (160 ngày) và tối đa 84 ngày lịch sử trước mỗi cửa sổ để lập đặc trưng. Baseline lấy trung bình 4 quan sát gần nhất cùng thứ trên thang log. Gradient boosting và MLP hai tầng ẩn học phần sai lệch của baseline. Lưới 5 trọng số trên đợt đầu chọn tỷ lệ 50/50 và cố định cho đợt thứ hai cùng file nộp.

| Giai đoạn 2017 | Baseline | Boosting | MLP | Kết hợp |
|---|---:|---:|---:|---:|
| 15–30/07, chọn trọng số | 0,4295 | 0,4005 | 0,4006 | 0,3964 |
| 31/07–15/08, kiểm định | 0,5398 | 0,4182 | 0,4247 | 0,4166 |

RMSLE càng thấp càng tốt. Cả ba lần fit MLP chạm giới hạn 32 epoch. Phân tích XAI trong slide đo mức tăng RMSLE khi hoán vị từng đặc trưng của dự báo boosting trên đợt kiểm định cuối. Ngày test là 16–31/08/2017. Điểm leaderboard cần được bổ sung sau khi nộp.
