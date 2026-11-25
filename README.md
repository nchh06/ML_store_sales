# Dự báo doanh số Favorita

File nộp là `submission.csv`. Báo cáo thuyết trình là `report.pptx`. Cả hai được tạo từ lần chạy local trên dữ liệu Favorita. Đề bài nằm trong `docs/Final_INT3405E_5.pdf`.

## Nộp trên Kaggle

1. Đăng nhập và tham gia [Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting). Theo đề, trưởng nhóm đặt tên đội theo mẫu `UET_[INT3405E_5]_[số thứ tự nhóm]`.
2. Trong mục **Submit Predictions**, chọn `submission.csv`. Nếu nộp từ notebook, import `store_sales.ipynb`, gắn dữ liệu cuộc thi, chọn **Save Version → Save & Run All**. Notebook xuất `/kaggle/working/submission.csv`.
3. Ghi nhận RMSLE do Kaggle trả về và cập nhật slide kết quả trước khi thuyết trình.

## Tái lập trên máy cá nhân

Dùng Python 3.12. Đặt `train.csv`, `test.csv`, `sample_submission.csv`, `stores.csv` và `holidays_events.csv` vào `data/`. Có thể giải nén năm file từ [trang dữ liệu Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data). Lần chạy local dùng [bản sao dữ liệu của t4tiana, revision 5768b1d](https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting/tree/5768b1d).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python forecast.py --data data --out .
python -m unittest -v test_forecast
python make_report.py
```

`forecast.py` xuất `results.json` và thay thế `submission.csv`. Dùng `--out` để chọn thư mục kết quả khác. `make_report.py` đọc `results.json` trong thư mục hiện tại và tạo lại báo cáo. Notebook chứa mã dự báo độc lập, dùng thư viện có sẵn của Kaggle. Để tái lập theo byte, dùng môi trường local đã ghim phiên bản. SHA-256 của file nộp đã kiểm tra là `3c15dfa1a11481e22a5be627e09d7988f94b826b6fbaae3961ead2cba2f3b1b6`.

## Kết quả kiểm định

Mô hình Ridge riêng cho từng cặp cửa hàng và nhóm hàng dùng tối đa 364 ngày lịch sử. Trọng số quan sát giảm một nửa sau 48 ngày. Lịch nghỉ phản ánh cả phạm vi địa lý và ngày nghỉ được điều chuyển. Boosting và MLP học phần chênh so với Ridge trên 12 cửa sổ 16 ngày liên tiếp gần nhất cùng 6 cửa sổ quanh cùng mùa năm trước. Mô hình có thể so sánh khuyến mãi hiện tại với lịch sử cùng thứ và với các cửa hàng bán cùng nhóm hàng.

Tỷ lệ kết hợp là 50% boosting và 50% MLP trên thang log, được chọn ở đợt 15–30/07/2017 rồi giữ cố định cho file nộp. Bảng dưới so sánh RMSLE của hai phiên bản trên cùng bộ nhãn. Điểm thấp hơn tốt hơn.

| Cửa sổ 16 ngày | Bản trước (c4b504c) | Bản mới |
|---|---:|---:|
| 16–31/08/2016, đối chiếu cùng mùa | 0,7801 | 0,6934 |
| 29/06–14/07/2017, đối chiếu thêm | 0,3795 | 0,3704 |
| 15–30/07/2017, chọn trọng số | 0,3964 | 0,3930 |
| 31/07–15/08/2017, so sánh cuối | 0,4166 | 0,3977 |

Các mốc lịch sử dùng để so sánh hồi cứu. Hai mốc cuối đã từng được xem ở phiên bản trước. Mỗi lần fit chỉ sử dụng nhãn trước ngày bắt đầu dự báo tương ứng. Điểm từng thành phần, sai số theo nhóm hàng và kết quả hoán vị đặc trưng nằm trong `results.json`. MLP chạm giới hạn 32 epoch trong lần chạy này. Điểm leaderboard cần được ghi nhận sau khi nộp file cho 16–31/08/2017.

Bản nộp trước được lưu trong lịch sử Git. Có thể xuất riêng bằng `git show c4b504c:submission.csv > submission_previous.csv`.
