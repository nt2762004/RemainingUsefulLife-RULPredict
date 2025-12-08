# GIẢI THÍCH CHI TIẾT NOTEBOOK VÀ EDA

Tài liệu này giải thích chi tiết các bước thực hiện trong các Jupyter Notebook của dự án, tập trung vào quy trình Khám phá dữ liệu (EDA) và Xây dựng mô hình (Modeling).

> **Lưu ý:** Để xem hướng dẫn chạy code (Train/Predict), xem file [README_HowToRun.md](README_HowToRun.md).

## 1. `eda_preprocessing.ipynb` (Khám phá và Tiền xử lý dữ liệu)

Notebook này thực hiện các bước chuẩn bị dữ liệu quan trọng trước khi đưa vào mô hình.

### Bước 1: Khám phá dữ liệu (EDA)
- **Load dữ liệu**: Đọc dữ liệu thô từ `data/raw/Battery_RUL.csv`.
- **Trực quan hóa**:
  - *Phân bố RUL*: Xem histogram để hiểu phân phối của biến mục tiêu.
  - *RUL theo chu kỳ*: Vẽ biểu đồ đường để thấy xu hướng giảm dần của RUL theo thời gian (Cycle Index).
  - *Ma trận tương quan (Heatmap)*: Xem mối quan hệ giữa các đặc trưng (features) với nhau và với RUL.

### Bước 2: Tiền xử lý và Làm sạch dữ liệu
- **Xử lý giá trị âm/bằng 0**: Các cột thời gian (Time columns) không thể có giá trị <= 0, được chuyển thành NaN.
- **Xử lý ngoại lai toàn cục (Global Outliers)**: Loại bỏ các giá trị "siêu lớn" vô lý (ví dụ: thời gian xả quá dài do lỗi cảm biến) bằng cách dùng quantile 99.5%.
- **Lọc nhiễu cục bộ (Local Outliers)**: Sử dụng phương pháp Rolling Median (cửa sổ trượt) để phát hiện các điểm dữ liệu nhảy vọt bất thường so với các điểm lân cận và loại bỏ chúng.
- **Điền dữ liệu (Interpolation)**: Sử dụng nội suy tuyến tính (linear interpolation) để điền vào các giá trị NaN đã tạo ra ở các bước trên, giúp dữ liệu mượt mà và liên tục theo chuỗi thời gian.

### Bước 3: Lưu kết quả
- Dữ liệu sạch được lưu vào `data/processed/Battery_RUL_processed.csv` để dùng cho các bước modeling sau này.

---

## 2. `battery_rul_modelingv2.ipynb` (Huấn luyện mô hình cải tiến)

Đây là notebook chính để xây dựng và đánh giá mô hình, với các cải tiến để tránh overfitting.

### Bước 1: Import thư viện
- Sử dụng `pandas`, `numpy` để xử lý dữ liệu.
- `sklearn` cho các mô hình (LinearRegression, RandomForest) và đánh giá (RMSE, MAE, R2).

### Bước 2: Load dữ liệu
- Đọc file `data/processed/Battery_RUL_processed.csv`.
- Sắp xếp lại theo `Cycle_Index` để đảm bảo tính thứ tự thời gian cho việc tạo feature.

### Bước 3: Feature Engineering (Tạo đặc trưng mới)
- **Efficiency_Ratio**: Tỷ lệ thời gian xả / sạc.
- **Voltage_Drop_Rate**: Tốc độ sụt áp trung bình ((Max Voltage - Min Voltage) / Discharge Time).
- **Discharge_Drop_Rate**: Tốc độ giảm thời gian xả so với chu kỳ trước (dùng `.diff()`).
- **Rolling Mean/Std (window=10)**: Trung bình và độ lệch chuẩn trượt của 10 chu kỳ gần nhất (làm mượt dữ liệu và bắt xu hướng biến động).
- **Lag Feature (Prev_Cycle_Discharge)**: Giá trị thời gian xả của chu kỳ trước (tính chất Markov - trạng thái hiện tại phụ thuộc quá khứ gần).
- **Xử lý dữ liệu lỗi**: Loại bỏ các dòng có `Voltage_Drop_Rate < 0` (vô lý vật lý).

### Bước 4: Chia tập dữ liệu (Train/Test Split)
- **QUAN TRỌNG**: Loại bỏ cột `Cycle_Index` khỏi tập features (X). Việc này ngăn mô hình "học vẹt" (chỉ nhớ số thứ tự chu kỳ để đoán RUL) và bắt buộc mô hình phải học từ các đặc trưng vật lý của pin.
- Chia train/test theo tỷ lệ 80/20.
- Chuẩn hóa dữ liệu (StandardScaler) để đưa các feature về cùng quy mô.

### Bước 5: Huấn luyện và đánh giá mô hình
- **Linear Regression**: Mô hình cơ bản để làm baseline so sánh.
- **Random Forest Regressor**: Mô hình ensemble mạnh mẽ, thường cho kết quả tốt với dữ liệu bảng.
- **XGBoost**: Mô hình Gradient Boosting tối ưu, thường đạt hiệu suất cao nhất trên các bài toán dạng bảng.
- **Deep Learning Models (LSTM, GRU, CNN)**: Các mô hình mạng nơ-ron sâu để bắt các đặc trưng chuỗi thời gian phức tạp (nếu dữ liệu được xử lý dưới dạng sequence).
- **Đánh giá**: Sử dụng RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), và R2 Score. Trực quan hóa so sánh Actual vs Predicted và phân tích Residuals.

---

## 3. `battery_rul_modelingv1.ipynb` (Phiên bản thử nghiệm)

- Phiên bản đầu tiên, thử nghiệm các mô hình cơ bản.
- Có thể vẫn giữ feature `Cycle_Index`, dẫn đến kết quả dự đoán có thể rất cao nhưng không thực tế (overfitting) khi áp dụng cho pin mới hoặc điều kiện vận hành khác.
