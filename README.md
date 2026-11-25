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

`forecast.py` xuất `results.json` và thay thế `submission.csv`. Dùng `--out` để chọn thư mục kết quả khác. `make_report.py` đọc `results.json` trong thư mục hiện tại và tạo lại báo cáo. Notebook chứa mã dự báo độc lập, dùng thư viện có sẵn của Kaggle. Để tái lập theo byte, dùng môi trường local đã ghim phiên bản. SHA-256 của file nộp đã kiểm tra là `c3e4a3ee66f2d267d6e10cafe27b689838a5a67327fbeac41b1b51fb775731a0`.

## Kết quả kiểm định

Mô hình Ridge riêng cho từng cặp cửa hàng và nhóm hàng dùng tối đa 364 ngày lịch sử. Trọng số quan sát giảm một nửa sau 48 ngày. Lịch nghỉ phản ánh cả phạm vi địa lý và ngày nghỉ được điều chuyển. Boosting và MLP học phần chênh so với Ridge trên 12 cửa sổ 16 ngày liên tiếp gần nhất cùng 6 cửa sổ quanh cùng mùa năm trước. Mô hình có thể so sánh khuyến mãi hiện tại với lịch sử cùng thứ và với các cửa hàng bán cùng nhóm hàng.

LightGBM huấn luyện thêm một mô hình cho mỗi nhóm hàng trên tối đa 900 ngày. Nhánh này dự báo trực tiếp log-doanh số từ đặc trưng theo ngày chốt. Hai nhánh Ridge đã hiệu chỉnh chiếm 50% và LightGBM trực tiếp chiếm 50% của tổ hợp. Khi một cửa hàng có ít nhất ba ngày liên tiếp với tổng số mặt hàng khuyến mãi bằng 0, các ngày đó dùng dự báo Ridge đã hiệu chỉnh. Quy tắc dự phòng này ảnh hưởng đến cửa sổ tháng 8/2016, không kích hoạt trong test tháng 8/2017.

Ngày 16/08/2017, số mặt hàng khuyến mãi tăng đồng loạt trên nhiều nhóm hàng. Hàm `robust_promotions` so sánh phân vị 20 giữa các cửa hàng với các ngày cùng thứ trong 84 ngày trước và trong cửa sổ dự báo. Khi ít nhất 20% nhóm hàng (tối thiểu bốn nhóm) vượt ngưỡng, dùng 90% trung vị khuyến mãi của các ngày cùng thứ còn lại trong cửa sổ và 10% mức được cung cấp. Chỉ 16/08/2017 kích hoạt quy tắc trong test. Đây là giả định về độ tin cậy của tín hiệu khuyến mãi. Doanh số thật của ngày này thuộc tập Kaggle chấm.

| Cửa sổ 16 ngày | Phiên bản b898b96 | Phiên bản hiện tại |
|---|---:|---:|
| 16–31/08/2016, cùng mùa | 0,69335 | 0,68969 |
| 13–28/06/2017 | 0,38424 | 0,37865 |
| 15–30/07/2017, chọn tỷ lệ cho hai nhánh Ridge | 0,39296 | 0,39194 |
| 31/07–15/08/2017 | 0,39775 | 0,39004 |

Bảng so sánh sử dụng cùng các nhãn lịch sử. Các mốc đã được xem trong quá trình phát triển. Phép thử độ nhạy riêng trên 31/07/2017 can thiệp giả lập vào số khuyến mãi tại mọi cửa hàng. RMSLE của nhánh cũ trong ngày đó là 0,39074 với dữ liệu gốc, 1,05588 với cú sốc giả lập và 0,43260 sau xử lý. Phép thử chỉ đánh giá phản ứng với đầu vào biến động.

Public Score do người nộp ghi nhận cho bản trước là **0,44587**. Điểm của bản hiện tại cần cập nhật từ Kaggle sau khi nộp. Dùng `git show b898b96:submission.csv > submission_previous.csv` để xuất lại bản trước.
