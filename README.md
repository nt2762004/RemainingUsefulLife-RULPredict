# Battery RUL Prediction Project (English Version)

This project builds a Machine Learning system to predict the **Remaining Useful Life (RUL)** of Li-ion batteries based on charge/discharge cycle data.

The system focuses on processing time-series data, extracting important physical features of the battery, and applying machine learning models from basic to advanced to give accurate predictions.

## Folder Structure

```
├── eda_preprocessing.ipynb       # Notebook for EDA and data preprocessing (filtering noise, cleaning)
├── battery_rul_modelingv2.ipynb  # Main Notebook: Feature Engineering and model training (RF, XGBoost, DL)
├── battery_rul_modelingv1.ipynb  # Initial test Notebook (Baseline)
├── train.py                      # Training script for standard pipeline (for production)
├── predict.py                    # Script to predict RUL for new data
├── evaluate.py                   # Script to evaluate model performance on test set
├── configs/                      # Folder containing config files (config.yaml)
├── data/                         # Data folder
│   ├── raw/                      # Raw data (Battery_RUL.csv)
│   └── processed/                # Processed data (Battery_RUL_processed.csv)
├── models/                       # Folder to save trained models (.joblib)
└── outputs/                      # Folder containing prediction results (.csv)
```

> **Note:** To see instructions on how to run the code (Train/Predict), see the file [README_HowToRun.md](README_HowToRun.md).

### 1. `eda_preprocessing.ipynb` (Exploration and Preprocessing)
This notebook performs important data preparation steps before putting data into the model.

*   **Goal:** Understand data characteristics, remove noise, and create a clean dataset.
*   **Process:**
    *   **Load Data:** Read raw data from `data/raw/Battery_RUL.csv`.
    *   **Visualization (EDA):** Draw charts for RUL distribution, RUL decrease trend by cycle, and correlation matrix (Heatmap).
    *   **Handle Error Data (Data Cleaning):**
        *   *Physical Logic:* Time columns (like charge/discharge time) cannot be negative or zero. These values are changed to `NaN`.
        *   *Global Outliers:* Remove "very large" values caused by sensor errors (using 99.5% quantile threshold to cut off extreme values).
    *   **Smooth Data (Smoothing & Local Outliers):**
        *   Use **Rolling Median** method to find and remove local noise points (sudden jumps compared to neighbors in the time series).
    *   **Fill Missing Data (Imputation):**
        *   Use **Linear Interpolation** to fill `NaN` values created in previous steps. This method helps restore time series continuity better than filling with the average.
    *   **Result:** Save file `data/processed/Battery_RUL_processed.csv`.

### 2. `battery_rul_modelingv2.ipynb` (Improved Model Training)
Main version.

*   **Goal:** Build RUL prediction model avoiding "rote learning" (overfitting to cycle index).
*   **Process:**
    *   **Feature Engineering:**
        *   *Physical:* Efficiency Ratio (Discharge/Charge ratio), Voltage Drop Rate.
        *   *Statistical:* Rolling Mean/Std (Trend and stability in last 10 cycles).
        *   *Time Series:* Lag features (Values of previous cycle).
    *   **Prepare Data:** Remove `Cycle_Index` column to force the model to learn from battery features instead of order number. Split Train/Test (80/20) and normalize (StandardScaler).
    *   **Train Model:**
        *   *Machine Learning:* Linear Regression (Baseline), Random Forest, XGBoost.
        *   *Deep Learning:* LSTM, GRU, CNN (for sequence data).
    *   **Evaluate:** Use RMSE, MAE, R2 Score and analyze Residuals charts.

### 3. `battery_rul_modelingv1.ipynb` (Initial Test)
Test version.

*   **Goal:** Run quick tests of basic models.
*   **Characteristics:** Keeps `Cycle_Index` feature, leading to prediction results that might be very high on test set with same distribution but poor in real application (overfitting). Used to compare with v2.

### 4. Scripts (`train.py`, `predict.py`, `evaluate.py`)
Pure Python source code to run automated pipeline (MLOps).

*   **`train.py`:** Automatically reads config, loads data, trains and saves model to `models/` folder.
*   **`evaluate.py`:** Loads saved model and evaluates again on test set.
*   **`predict.py`:** Receives input CSV file (new battery data), runs through preprocessing steps similar to v2 and exports prediction results to `outputs/predictions.csv`.

## Installation Requirements

The project requires these Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib pyyaml
```
---

# Dự án Dự đoán Tuổi thọ Pin (Battery RUL Prediction) (Vietnamese Version)

Dự án này xây dựng một hệ thống Machine Learning để dự đoán **tuổi thọ còn lại (Remaining Useful Life - RUL)** của pin Li-ion dựa trên dữ liệu chu kỳ sạc/xả.

Hệ thống tập trung vào việc xử lý dữ liệu chuỗi thời gian, trích xuất các đặc trưng vật lý quan trọng của pin và áp dụng các mô hình học máy từ cơ bản đến nâng cao để đưa ra dự đoán chính xác.

## Cấu trúc Thư mục

```
├── eda_preprocessing.ipynb       # Notebook khám phá (EDA) và tiền xử lý dữ liệu (lọc nhiễu, làm sạch)
├── battery_rul_modelingv2.ipynb  # Notebook chính: Feature Engineering và huấn luyện mô hình (RF, XGBoost, DL)
├── battery_rul_modelingv1.ipynb  # Notebook thử nghiệm ban đầu (Baseline)
├── train.py                      # Script huấn luyện pipeline chuẩn (cho production)
├── predict.py                    # Script dự đoán RUL cho dữ liệu mới
├── evaluate.py                   # Script đánh giá hiệu năng mô hình trên tập test
├── configs/                      # Thư mục chứa file cấu hình (config.yaml)
├── data/                         # Thư mục dữ liệu
│   ├── raw/                      # Dữ liệu thô (Battery_RUL.csv)
│   └── processed/                # Dữ liệu đã qua xử lý (Battery_RUL_processed.csv)
├── models/                       # Thư mục lưu các model đã huấn luyện (.joblib)
└── outputs/                      # Thư mục chứa kết quả dự đoán (.csv)
```

> **Lưu ý:** Để xem hướng dẫn chạy code (Train/Predict) xem file [README_HowToRun.md](README_HowToRun.md).

### 1. `eda_preprocessing.ipynb` (Khám phá và Tiền xử lý)
Notebook này thực hiện các bước chuẩn bị dữ liệu quan trọng trước khi đưa vào mô hình.

*   **Mục tiêu:** Hiểu đặc điểm dữ liệu, loại bỏ nhiễu và tạo ra bộ dữ liệu sạch.
*   **Quy trình:**
    *   **Load dữ liệu:** Đọc dữ liệu thô từ `data/raw/Battery_RUL.csv`.
    *   **Trực quan hóa (EDA):** Vẽ biểu đồ phân bố RUL, xu hướng giảm RUL theo chu kỳ, và ma trận tương quan (Heatmap).
    *   **Xử lý dữ liệu lỗi (Data Cleaning):**
        *   *Logic vật lý:* Các cột thời gian (như thời gian sạc/xả) không thể có giá trị âm hoặc bằng 0. Các giá trị này được chuyển thành `NaN`.
        *   *Xử lý ngoại lai toàn cục (Global Outliers):* Loại bỏ các giá trị "siêu lớn" vô lý do lỗi cảm biến (sử dụng ngưỡng quantile 99.5% để cắt bỏ phần đuôi phân phối cực đoan).
    *   **Làm mượt dữ liệu (Smoothing & Local Outliers):**
        *   Sử dụng phương pháp **Rolling Median** (Trung vị trượt) để phát hiện và loại bỏ các điểm dữ liệu nhiễu cục bộ (nhảy vọt bất thường so với các điểm lân cận trong chuỗi thời gian).
    *   **Điền dữ liệu khuyết thiếu (Imputation):**
        *   Sử dụng **Nội suy tuyến tính (Linear Interpolation)** để điền vào các giá trị `NaN` đã tạo ra ở các bước trên. Phương pháp này giúp khôi phục tính liên tục của chuỗi thời gian tốt hơn so với điền bằng trung bình.
    *   **Kết quả:** Lưu file `data/processed/Battery_RUL_processed.csv`.

### 2. `battery_rul_modelingv2.ipynb` (Huấn luyện Mô hình Cải tiến)
Phiên bản chính.

*   **Mục tiêu:** Xây dựng mô hình dự đoán RUL tránh hiện tượng "học vẹt" (overfitting vào chỉ số chu kỳ).
*   **Quy trình:**
    *   **Feature Engineering (Tạo đặc trưng):**
        *   *Vật lý:* Efficiency Ratio (Tỷ lệ xả/sạc), Voltage Drop Rate (Tốc độ sụt áp).
        *   *Thống kê:* Rolling Mean/Std (Xu hướng và độ ổn định trong 10 chu kỳ gần nhất).
        *   *Chuỗi thời gian:* Lag features (Giá trị của chu kỳ trước).
    *   **Chuẩn bị dữ liệu:** Loại bỏ cột `Cycle_Index` để ép mô hình học từ đặc trưng pin thay vì số thứ tự. Chia tập Train/Test (80/20) và chuẩn hóa (StandardScaler).
    *   **Huấn luyện Mô hình:**
        *   *Machine Learning:* Linear Regression (Baseline), Random Forest, XGBoost.
        *   *Deep Learning:* LSTM, GRU, CNN (cho dữ liệu chuỗi).
    *   **Đánh giá:** Sử dụng RMSE, MAE, R2 Score và phân tích biểu đồ Residuals.

### 3. `battery_rul_modelingv1.ipynb` (Thử nghiệm Ban đầu)
Phiên bản thử nghiệm.

*   **Mục tiêu:** Chạy thử nghiệm nhanh các mô hình cơ bản.
*   **Đặc điểm:** Giữ feature `Cycle_Index`, dẫn đến kết quả dự đoán có thể rất cao trên tập test cùng phân phối nhưng kém khi áp dụng thực tế (overfitting). Dùng để so sánh với v2.

### 4. Các Scripts (`train.py`, `predict.py`, `evaluate.py`)
Bộ mã nguồn Python thuần để chạy pipeline tự động hóa (MLOps).

*   **`train.py`:** Tự động đọc config, load dữ liệu, huấn luyện và lưu model vào thư mục `models/`.
*   **`evaluate.py`:** Load model đã lưu và đánh giá lại trên tập test.
*   **`predict.py`:** Nhận file CSV đầu vào (dữ liệu pin mới), chạy qua bước tiền xử lý tương tự v2 và xuất kết quả dự đoán ra `outputs/predictions.csv`.

## Yêu cầu cài đặt

Dự án yêu cầu các thư viện Python sau:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib pyyaml
```
