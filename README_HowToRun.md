HƯỚNG DẪN CHẠY DỰ ÁN Battery RUL
=================================

Mục tiêu
--------
Dự án cung cấp pipeline thuần Python để huấn luyện (train), đánh giá (evaluate) và suy luận (predict) tuổi thọ còn lại (RUL) của pin từ dữ liệu bảng CSV.

Mặc định sử dụng dữ liệu đã xử lý: data/processed/Battery_RUL_processed.csv


Cấu trúc chính
--------------
- configs/config.yaml
  • File cấu hình trung tâm: đường dẫn dữ liệu, random_state, test_size, tham số model.

- train.py
  • Script HUẤN LUYỆN chính: đọc dữ liệu, tách features/target (RUL), chia train/test, huấn luyện Pipeline [StandardScaler + RandomForest], đánh giá nhanh, lưu model .joblib và metadata .meta.json vào thư mục models/.

- evaluate.py (tùy chọn)
  • Nạp model đã lưu và chấm lại trên test split (theo config). In RMSE/MAE/R2.

- predict.py
  • Nạp model đã lưu và dự đoán RUL cho 1 file CSV đầu vào. Nếu input có cột RUL thì sẽ tự bỏ qua khi dự đoán. Ghi kết quả ra outputs/predictions.csv.

- Thư mục rul/ (thư viện nội bộ, không chạy trực tiếp)
  • rul/config.py: Dataclass cấu hình + load_config.
  • rul/data.py: read_csv, get_features_and_target, split_train_test.
  • rul/model.py: xây dựng Pipeline (tiền xử lý + RandomForestRegressor).
  • rul/metrics.py: RMSE, MAE, R2, regression_report.
  • rul/utils.py: logging, tạo thư mục, lưu JSON, timestamp.

- data/
  • processed/Battery_RUL_processed.csv (mặc định được dùng để train/eval/predict)
  • raw/Battery_RUL.csv (dữ liệu gốc — chỉ dùng khi muốn override)

- models/
  • Nơi lưu các file model .joblib và metadata .meta.json khi train xong.

- outputs/
  • Nơi lưu các file dự đoán ví dụ outputs/predictions.csv.

- Notebook (tùy chọn tham khảo, không bắt buộc để pipeline chạy)
  • eda_preprocessing.ipynb: khám phá dữ liệu, tiền xử lý, trực quan.
  • battery_rul_modelingv1.ipynb: thử nghiệm mô hình cơ bản.
  • battery_rul_modelingv2.ipynb: phiên bản cải tiến, loại bỏ feature 'Cycle_Index' để tránh mô hình học vẹt (overfitting vào chỉ số chu kỳ), bổ sung feature engineering.


Thứ tự chạy chuẩn
-----------------
1) Train (bắt buộc)
   PowerShell:
   
   python .\train.py
   
   • Kết quả: tạo models\rul_model_YYYYMMDD-HHMMSS.joblib và models\rul_model_YYYYMMDD-HHMMSS.meta.json

2) Evaluate (tùy chọn)
   PowerShell:
   
   python .\evaluate.py --model ".\models\rul_model_YYYYMMDD-HHMMSS.joblib"

3) Predict (suy luận)
   PowerShell:
   
   python .\predict.py --model ".\models\rul_model_YYYYMMDD-HHMMSS.joblib" --input_csv "data\processed\Battery_RUL_processed.csv" --output_csv "outputs\predictions.csv"


Tùy chỉnh nhanh
----------------
- Đổi đường dẫn dữ liệu hoặc tham số mô hình: sửa trong configs/config.yaml.
- Hoặc override khi chạy:
  • Train với dữ liệu khác (ví dụ file raw):
    python .\train.py --data "data\raw\Battery_RUL.csv"

  • Evaluate trên dữ liệu khác:
    python .\evaluate.py --model ".\models\rul_model_....joblib" --data "data\processed\Battery_RUL_processed.csv"

  • Predict trên CSV khác:
    python .\predict.py --model ".\models\rul_model_....joblib" --input_csv "path\to\your.csv" --output_csv "outputs\predictions.csv"


Vai trò notebook vs. scripts
----------------------------
- Notebook (.ipynb): nơi khám phá/trực quan/thử nghiệm cho phần trình bày.
- Scripts (.py): quy trình chạy chuẩn để huấn luyện/đánh giá/suy luận có thể lặp lại và tái sử dụng.
