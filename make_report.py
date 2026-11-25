"""Tạo slide từ các kết quả đã đo bằng forecast.py."""
import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.util import Inches, Pt

r = json.loads(Path("results.json").read_text())
e = r["eda"]
fold1, fold2 = r["folds"]
w = r["tree_weight"]
deck = Presentation()
deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)
ink, accent, muted = "18344A", "007C78", "4E6475"


def label(slide, x, y, width, height, content, size, color=ink, bold=False):
    box = slide.shapes.add_textbox(*[Inches(i) for i in (x, y, width, height)])
    box.text_frame.word_wrap = True
    for i, line in enumerate(content.splitlines()):
        p = box.text_frame.paragraphs[0] if i == 0 else box.text_frame.add_paragraph()
        p.text = line
        p.font.name = "DejaVu Sans"
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = RGBColor.from_string(color)
    return box


def slide(section, title):
    page = deck.slides.add_slide(deck.slide_layouts[6])
    label(page,.55,.35,12,.4,section.upper(),13,accent,True)
    label(page,.55,.95,12.25,.9,title,34,ink,True)
    label(page,.55,7,12,.3,f"INT3405  /  Store Sales – Time Series Forecasting                                     {len(deck.slides):02d}",10,muted)
    return page


def block(page, x, y, width, title, body):
    label(page,x,y,width,.55,title,22,accent,True)
    label(page,x,y+.65,width,2,body,18)


def chart(page, categories, series, kind, height=4.1, fmt="0.00"):
    data = CategoryChartData()
    data.categories = categories
    for name, values in series.items():
        data.add_series(name, values)
    c = page.shapes.add_chart(kind, Inches(.7), Inches(2), Inches(11.85), Inches(height), data).chart
    c.has_legend = len(series) > 1
    if c.has_legend:
        c.legend.position = XL_LEGEND_POSITION.BOTTOM
        c.legend.include_in_layout = False
    c.value_axis.tick_labels.number_format = fmt
    c.category_axis.tick_labels.font.size = Pt(12)
    for s, color in zip(c.series, [accent, "DC8245"]):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = RGBColor.from_string(color)
        s.format.line.color.rgb = RGBColor.from_string(color)
    return c


s = slide("Bài tập lớn Học máy", "Dự báo 16 ngày bán hàng tại Favorita")
label(s,.62,2.1,12,1.25,"54 cửa hàng × 33 nhóm sản phẩm tại Ecuador",29)
label(s,.62,3.5,12,1.3,"Mỗi ngày cần dự báo 1.782 giá trị doanh số. Mỗi đợt nộp gồm 28.512 dự báo, kể cả các chuỗi có nhiều ngày bằng 0.",22)
label(s,.62,5.65,12,.7,f"RMSLE của phương án nộp trên đợt kiểm định cuối    {fold2['scores']['Blend']:.4f}",22,accent,True)

s = slide("Phân tích dữ liệu", "Mức bán thay đổi theo từng tuần")
weekly = e["weekly_sales"]
chart(s,[d[5:] for d in weekly],{"Triệu đơn vị mỗi ngày":[x/1e6 for x in weekly.values()]},XL_CHART_TYPE.LINE_MARKERS,3.55)
label(s,.7,5.7,12,1,"Tổng doanh số của chuỗi siêu thị, lấy trung bình theo các ngày quan sát trong tuần. Tuần cuối chỉ có dữ liệu đến 15/08.",17,muted)

s = slide("Tiền xử lý", "Giữ nguyên ngày không bán hàng")
block(s,.65,2,5.8,"Lịch sử quan sát",f"{e['train_rows']:,} dòng từ {e['start']} đến {e['end']}. Có {e['zero_share']:.1%} dòng ghi nhận doanh số bằng 0.")
block(s,6.65,2,5.8,"Căn chỉnh trục ngày",f"Lập lịch liên tục theo từng cặp cửa hàng và nhóm hàng. {e['missing_dates']} ngày vắng trong train được giữ là thiếu khi tính trung bình.")
block(s,.65,4.45,5.8,"Biến mục tiêu","Dùng log(1 + sales) để giữ giá trị 0 và giảm ảnh hưởng của chênh lệch quy mô giữa các nhóm.")
block(s,6.65,4.45,5.8,"Dữ liệu tại mốc dự báo","Lịch và số mặt hàng khuyến mãi đã có cho ngày đích. Mọi đặc trưng doanh số chỉ dùng ngày trước mốc dự báo.")

s = slide("Phương pháp", "Hai mô hình cùng hiệu chỉnh dự báo theo thứ")
label(s,.65,1.95,12,.85,"Baseline là trung bình log-doanh số của 4 ngày cùng thứ gần nhất tại cùng cửa hàng và nhóm hàng.",20)
block(s,.65,3.05,5.8,"Gradient boosting","110 vòng boosting trên histogram, tối đa 31 lá mỗi cây. Các cây cộng dồn phần hiệu chỉnh cho baseline.")
block(s,6.65,3.05,5.8,"Mạng MLP","Chuẩn hóa đặc trưng, dùng hai tầng ẩn ReLU gồm 48 và 24 nút. Adam học phần chênh so với baseline, tối đa 32 epoch.")
label(s,.65,5.65,12,1,"Cả hai dùng mức bán 7/28/84 ngày, lịch sử cùng thứ, tần suất bằng 0, khuyến mãi, cửa hàng, nhóm hàng và lịch.",18,muted)

s = slide("Thiết kế thực nghiệm", "Mô phỏng đúng thời điểm phải dự báo")
for i,(title,body) in enumerate([
    ("15–30/07  ·  Chọn trọng số","Thử 5 tỷ lệ boosting từ 0% đến 100% trên thang log, chọn tỷ lệ cho RMSLE nhỏ nhất."),
    ("31/07–15/08  ·  Kiểm định","Huấn luyện lại với dữ liệu mới, giữ trọng số đã chọn. So sánh với baseline và từng mô hình."),
    ("16–31/08  ·  Dự báo test","Dịch mốc đến 16/08. Huấn luyện lại trên 10 đợt lịch sử gần nhất, mỗi đợt dài 16 ngày.")]):
    label(s,.65,2+i*1.45,4.7,.95,title,20,accent,True)
    label(s,5.45,2+i*1.45,7.1,1.2,body,19)
label(s,.65,6.45,12,.45,"RMSLE = căn bậc hai của trung bình bình phương sai số trên thang log(1 + sales).",15,muted)

s = slide("Kết quả", "Sai số trên hai đợt dự báo liên tiếp")
names = list(fold1["scores"])
c = chart(s,names,{f["start"]:[f["scores"][k] for k in names] for f in r["folds"]},XL_CHART_TYPE.COLUMN_CLUSTERED,3.7,"0.000")
c.value_axis.minimum_scale = 0
c.plots[0].has_data_labels = True
c.plots[0].data_labels.number_format = "0.000"
c.plots[0].data_labels.font.size = Pt(11)
delta = 1-fold2["scores"]["Blend"]/fold2["scores"]["Baseline"]
label(s,.65,5.95,12,.9,f"Trọng số đã chọn: boosting {w:.0%}, MLP {1-w:.0%}. So với baseline, RMSLE giảm {delta:.1%} trên đợt kiểm định cuối.",18)

s = slide("XAI  /  Permutation importance", "Boosting nhạy với đặc trưng nào?")
top = sorted(r["permutation_delta"].items(), key=lambda a:a[1], reverse=True)[:5][::-1]
c = chart(s,[k for k,v in top],{"Mức tăng RMSLE":[v for k,v in top]},XL_CHART_TYPE.BAR_CLUSTERED,3.5,"0.000")
label(s,.65,5.65,12,1.2,f"Hoán vị từng cột trên đợt cuối 3 lần rồi đo lại toàn bộ dự báo boosting, gồm cả baseline. Đảo {top[-1][0]} làm RMSLE tăng trung bình {top[-1][1]:.3f}. Kết quả đo độ phụ thuộc dự báo và chịu ảnh hưởng của tương quan giữa các cột.",16)

s = slide("Phân tích sai số", "Những nhóm hàng khó dự báo nhất")
top = sorted(r["family_error"].items(),key=lambda a:a[1], reverse=True)[:5][::-1]
chart(s,[k for k,v in top],{"RMSLE":[v for k,v in top]},XL_CHART_TYPE.BAR_CLUSTERED,3.5,"0.00")
daily = r["daily_error"]
label(s,.65,5.65,12,1.15,f"Trên đợt cuối, nhóm {top[-1][0]} có RMSLE cao nhất ({top[-1][1]:.3f}). Theo ngày, RMSLE dao động từ {min(daily):.3f} đến {max(daily):.3f}. Hai cửa sổ mùa hè chỉ phản ánh một phần biến động trong năm.",17)

s = slide("Kết luận", "Từ kiểm định đến file nộp")
block(s,.65,2,5.85,"Phương án nộp",f"Kết hợp hai dự báo log theo tỷ lệ {w:.0%} / {1-w:.0%}. Đổi ngược bằng expm1, ghép đúng id của test với thứ tự của mẫu nộp.")
block(s,6.65,2,5.85,"Đo lường",f"RMSLE kiểm định cuối là {fold2['scores']['Blend']:.4f}. Điểm leaderboard do Kaggle tính sau khi nhóm nộp.")
label(s,.65,4.85,12,1.4,"Các bước tái lập và phiên bản thư viện có trong README. Dữ liệu gốc thuộc cuộc thi Kaggle Store Sales – Time Series Forecasting của Corporación Favorita.",19)
deck.save("report.pptx")
