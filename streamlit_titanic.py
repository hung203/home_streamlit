import datetime
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

st.title("Tiền xử lý dữ liệu Titanic cho Multiple Regression")

tab1, tab2, tab3,tab4 = st.tabs([
    "Xử lý dữ liệu",
    "Huấn luyện",
    "Dự đoán",
    "Mlflow"
])

# --------------------- Tab 1: Xử lý dữ liệu ---------------------
with tab1:
    st.header("1. Tải dữ liệu")
    # Sử dụng file uploader nếu có, ngược lại dùng file mặc định
    if "df" not in st.session_state:
        uploaded_file = st.file_uploader("Tải lên file CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv("titanic.csv")
        st.session_state.df = df.copy()
    else:
        df = st.session_state.df

    st.write("Dữ liệu ban đầu:")
    st.write(df)

    st.header("2. Xử lý giá trị thiếu")
    # Hiển thị bảng thống kê số lượng missing
    missing_data = df.isnull().sum().reset_index()
    missing_data.columns = ['Column', 'Missing Count']
    st.write("Bảng số lượng giá trị thiếu:")
    st.write(missing_data)

    # Tạo danh sách các cột có missing
    missing_cols = df.columns[df.isnull().any()].tolist()
    user_missing_choices = {}
    st.markdown("### Chọn phương pháp xử lý cho các cột có giá trị thiếu:")
    for col in missing_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Với cột số: cho phép fill với mean, median hoặc mode
            user_missing_choices[col] = st.selectbox(
                f"Phương pháp xử lý cho '{col}'", 
                options=["Giá trị trung bình", "Giá trị trung vị", "Giá trị xuất hiện nhiều nhất"],
                key=f"method_{col}"
            )
        else:
            # Với cột dạng chuỗi: chỉ cho fill với mode
            user_missing_choices[col] = st.selectbox(
                f"Phương pháp xử lý cho '{col}'", 
                options=["Giá trị xuất hiện nhiều nhất"],
                key=f"method_{col}"
            )

    if st.button("Xử lý giá trị thiếu"):
        for col, method in user_missing_choices.items():
            if pd.api.types.is_numeric_dtype(df[col]):
                if method == "Giá trị trung bình":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif method == "Giá trị trung vị":
                    df[col].fillna(df[col].median(), inplace=True)
                elif method == "Giá trị xuất hiện nhiều nhất":
                    df[col].fillna(df[col].mode()[0], inplace=True)    
            else:
                if method == "Giá trị xuất hiện nhiều nhất":
                    df[col].fillna(df[col].mode()[0], inplace=True)
        st.session_state.df = df.copy()
        st.session_state.missing_processed = True
        st.success("Xử lý giá trị thiếu thành công!")
        st.write("Dữ liệu sau khi xử lý giá trị thiếu:")
        st.write(st.session_state.df)
    else:
        st.info("Chưa thực hiện xử lý giá trị thiếu.")

    st.header("3. Mã hóa dữ liệu")
    if "encoded" not in st.session_state:
        if "Sex" in df.columns:
            df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
        if "Embarked" in df.columns:
            df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})
        st.session_state.df = df.copy()
        st.session_state.encoded = True
        st.success("Mã hóa dữ liệu thành công!")
        st.write("Dữ liệu sau khi mã hóa:")
        st.write(df)
    else:
        st.write("Dữ liệu sau khi mã hóa:")
        st.write(st.session_state.df)

    st.header("4. Xóa các cột không cần thiết")

    # Các cột mặc định sẽ được chọn
    default_cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]

    # Tạo danh sách chọn với giá trị mặc định
    selected_cols_to_drop = st.multiselect(
        "Chọn các cột muốn xóa:", 
        options=df.columns.tolist(), 
        default=[col for col in default_cols_to_drop if col in df.columns]
    )

    if st.button("Xóa các cột đã chọn"):
        if selected_cols_to_drop:
            df.drop(selected_cols_to_drop, axis=1, inplace=True)
            st.session_state.df = df.copy()
            st.success("Đã xóa các cột: " + ", ".join(selected_cols_to_drop))
        else:
            st.info("Không có cột nào được chọn.")

    st.write("Dữ liệu sau khi xóa các cột không cần thiết:")
    st.write(st.session_state.df)
    df = st.session_state.df  # đảm bảo df được cập nhật


    st.header("5. Chuẩn hóa dữ liệu")
    if st.button("Chuẩn hóa dữ liệu"):
        scaler = StandardScaler()
        default_numerical_features = ["Age", "Fare", "SibSp", "Parch", "Pclass", "Embarked"]
        numerical_features = [col for col in default_numerical_features if col in df.columns]
        if numerical_features:
            df[numerical_features] = scaler.fit_transform(df[numerical_features])
            st.session_state.df = df.copy()
            st.session_state.scaler = scaler
            st.success("Chuẩn hóa dữ liệu thành công!")
            st.write("Dữ liệu sau khi chuẩn hóa:")
            st.write(df)
        else:
            st.warning("Không có cột số nào để chuẩn hóa!")
    else:
        st.info("Nhấn nút 'Chuẩn hóa dữ liệu' để tiến hành chuẩn hóa.")

# --------------------- Tab 2: Huấn luyện ---------------------
with tab2:
    st.header("Chia dữ liệu")
    df = st.session_state.df  # Lấy dữ liệu đã xử lý từ session_state

    if "Survived" not in df.columns:
        st.error("Cột 'Survived' không tồn tại trong dữ liệu. Vui lòng không xóa cột này để tiến hành chia dữ liệu và huấn luyện mô hình.")
    else:
        # Cho phép người dùng chọn các biến (features) sử dụng cho mô hình (loại bỏ cột mục tiêu)
        X = df.drop("Survived", axis=1)  # Đặc trưng
        y = df["Survived"] 

        test_size = st.slider("Chọn tỉ lệ tập Test (%)", min_value=10, max_value=50, value=20, step=5) / 100.0
        valid_size = st.slider("Chọn tỉ lệ tập Valid (%) trên tập Train", min_value=10, max_value=50, value=20, step=5) / 100.0

        if st.button("Chia dữ liệu"):
            y = df["Survived"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=71)
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=71)
            
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.X_valid = X_valid
            st.session_state.y_valid = y_valid
            st.session_state.data_split_done = True
            
            st.success("Chia dữ liệu thành công!")
            st.write("Hình dạng của tập Train:", X_train.shape)
            st.write("Hình dạng của tập Valid:", X_valid.shape)
            st.write("Hình dạng của tập Test:", X_test.shape)
        if "data_split_done" not in st.session_state or not st.session_state.data_split_done:
            st.warning("Vui lòng bấm nút 'Chia dữ liệu' để tiến hành chia dữ liệu trước khi huấn luyện mô hình.")

    st.header("Huấn luyện & Kiểm thử mô hình")
    if "data_split_done" in st.session_state and st.session_state.data_split_done:
        algorithm = st.selectbox("Thuật toán:", ["Multiple Regression", "Polynomial Regression"])
        
        st.markdown("### Tùy chọn thông số của mô hình")
        if algorithm == "Multiple Regression":
            model = LinearRegression(fit_intercept=True)  # Luôn bật Intercept
        else:
            degree = st.number_input("Chọn bậc của đa thức:", min_value=2, max_value=5, value=2)

            # Bias luôn True
            poly_features = PolynomialFeatures(degree=degree, include_bias=True)
            linear_model = LinearRegression(fit_intercept=True)
            model = Pipeline([
                ('poly', poly_features),
                ('linear', linear_model)
            ])
        experiment_name = st.text_input(
            "Nhập tên cho thí nghiệm MLflow", 
            value="",
            help="Tên để lưu thí nghiệm trong MLflow."
        )
        if not experiment_name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_name = f"{algorithm}_{timestamp}"
        # Nút bấm để huấn luyện mô hình
        if st.button("Huấn luyện mô hình"):
            # Tạo tên thí nghiệm tự động dựa trên tên mô hình và thời gian hiện tại
            # Thiết lập tên thí nghiệm cho mlflow (nếu thí nghiệm chưa tồn tại, mlflow sẽ tạo mới)
            mlflow.set_experiment(experiment_name)
             # Lưu tên thí nghiệm vào session_state và hiển thị ra giao diện
            st.session_state.experiment_name = experiment_name
            st.write("Tên thí nghiệm:", experiment_name)
            with mlflow.start_run() as run:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                try:
                    scores = cross_val_score(model, st.session_state.X_train, st.session_state.y_train, 
                                            cv=cv, scoring='r2', error_score='raise')
                    
                    # Hiển thị Cross Validation Scores
                    cv_results_df = pd.DataFrame({
                        "Fold": [f"Fold {i+1}" for i in range(len(scores))],
                        "R² Score": scores
                    })
                    st.markdown("### Kết quả Cross Validation (R²)")
                    st.write(cv_results_df)
                    st.write("**R² trung bình:**", scores.mean())

                    mlflow.log_metric("cv_r2_mean", scores.mean())
                except Exception as e:
                    st.error("Lỗi khi chạy cross-validation: " + str(e))
                
                try:
                    model.fit(st.session_state.X_train, st.session_state.y_train)
                    y_pred = model.predict(st.session_state.X_test)
                    r2 = r2_score(st.session_state.y_test, y_pred)
                    mse = mean_squared_error(st.session_state.y_test, y_pred)
                    st.write("R-squared trên tập kiểm thử:", r2)
                    st.write("Mean Squared Error (MSE):", mse)
                    mlflow.log_metric("test_r2_score", r2)
                    mlflow.log_metric("test_MSE", mse)

                    # Hiển thị tham số của mô hình
                    st.markdown("### Tham số của mô hình")
                    if algorithm == "Multiple Regression":
                        coef_df = pd.DataFrame({
                            "Feature": st.session_state.X_train.columns,
                            "Coefficient": model.coef_
                        })
                        st.write(coef_df)
                        st.write("Hệ số chặn (Intercept):", model.intercept_)  # Vẫn hiển thị Intercept
                    else:
                        feature_names = poly_features.get_feature_names_out(st.session_state.X_train.columns)
                        coef_df = pd.DataFrame({
                            "Feature": feature_names,
                            "Coefficient": model.named_steps['linear'].coef_
                        })
                        st.write(coef_df)
                        st.write("Hệ số chặn của mô hình tuyến tính:", model.named_steps['linear'].intercept_)

                    st.session_state.trained_model = model
                except Exception as e:
                    st.error("Lỗi khi huấn luyện mô hình hoặc dự đoán: " + str(e))

    else:
        st.info("Chưa có dữ liệu được chia, vui lòng thực hiện bước chia dữ liệu.")




# --------------------- Tab 3: Dự đoán ---------------------
with tab3:
    st.header("Dự đoán sự sống sót")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3])
            sex = st.selectbox("Sex", ["male", "female"])
            age = st.slider("Age", 0, 100, 25)
        
        with col2:
            sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
            parch = st.number_input("Parents/Children", 0, 10, 0)
            fare = st.number_input("Fare", 0.0, 600.0, 50.0)
            embarked = st.selectbox("Embarked", ["C", "Q", "S"])
        
        if st.form_submit_button("Predict Survival"):
            # Kiểm tra xem mô hình đã được huấn luyện chưa
            if "trained_model" not in st.session_state:
                st.error("Mô hình chưa được huấn luyện. Vui lòng huấn luyện mô hình trước.")
            else:
                sex_encoded = 1 if sex == "female" else 0
                embarked_encoded = {"C": 0, "Q": 1, "S": 2}[embarked]
                input_df = pd.DataFrame({
                    "Pclass": [pclass],
                    "Sex": [sex_encoded],
                    "Age": [age],
                    "SibSp": [sibsp],
                    "Parch": [parch],
                    "Fare": [fare],
                    "Embarked": [embarked_encoded]
                })
                # Sử dụng scaler từ session_state
                if "scaler" in st.session_state:
                    scaler = st.session_state.scaler
                    cols_to_scale = ["Age", "Fare", "SibSp", "Parch", "Pclass", "Embarked"]
                    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
                else:
                    st.error("Chưa có scaler được lưu. Vui lòng chuẩn hóa dữ liệu trước.")
                
                try:
                    model = st.session_state.trained_model
                    prediction = model.predict(input_df)
                    predicted_class = 1 if prediction[0] >= 0.5 else 0
                    result = "🌟 Survived!" if predicted_class == 1 else "💀 Did Not Survive"
                    
                    # Kiểm tra xem input có tồn tại trong dữ liệu ban đầu hay không
                    features_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
                    matching_rows = st.session_state.df[features_cols].apply(
                        lambda row: np.allclose(row.values, input_df.iloc[0].values, atol=1e-6),
                        axis=1
                    )
                    
                    if matching_rows.any():
                        idx = matching_rows.idxmax()
                        actual_survived = st.session_state.df.loc[idx, "Survived"]
                        annotation = "Dự đoán đúng với thực tế" if actual_survived == predicted_class else "Dự đoán sai với thực tế"
                        st.subheader(f"Prediction Result: {result} ({annotation})")
                    else:
                        st.subheader(f"Prediction Result: {result} (Input không có trong bộ dữ liệu)")
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")
with tab4:
    st.header("5. Tracking MLflow")
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        # Lấy danh sách thí nghiệm từ MLflow
        experiments = mlflow.search_experiments()

        if experiments:
            st.write("#### Danh sách thí nghiệm")
            experiment_data = [
                {
                    "Experiment ID": exp.experiment_id,
                    "Experiment Name": exp.name,
                    "Artifact Location": exp.artifact_location
                }
                for exp in experiments
            ]
            df_experiments = pd.DataFrame(experiment_data)
            st.dataframe(df_experiments)

            # Chọn thí nghiệm dựa trên TÊN thay vì ID
            selected_exp_name = st.selectbox(
                "🔍 Chọn thí nghiệm để xem chi tiết",
                options=[exp.name for exp in experiments]
            )

            # Lấy ID tương ứng với tên được chọn
            selected_exp_id = next(exp.experiment_id for exp in experiments if exp.name == selected_exp_name)

            # Lấy danh sách runs trong thí nghiệm đã chọn
            runs = mlflow.search_runs(selected_exp_id)
            if not runs.empty:
                st.write("#### Danh sách runs")
                st.dataframe(runs)

                # Chọn run để xem chi tiết
                selected_run_id = st.selectbox(
                    "🔍 Chọn run để xem chi tiết",
                    options=runs["run_id"]
                )

                # Hiển thị chi tiết run
                run = mlflow.get_run(selected_run_id)
                st.write("##### Thông tin run")
                st.write(f"*Run ID:* {run.info.run_id}")
                st.write(f"*Experiment ID:* {run.info.experiment_id}")
                st.write(f"*Start Time:* {run.info.start_time}")

                # Hiển thị metrics
                st.write("##### Metrics")
                st.json(run.data.metrics)

                # Hiển thị params
                st.write("##### Params")
                st.json(run.data.params)

                # Hiển thị artifacts sử dụng client.list_artifacts
                artifacts = client.list_artifacts(selected_run_id)
                if artifacts:
                    st.write("##### Artifacts")
                    for artifact in artifacts:
                        st.write(f"- {artifact.path}")
            else:
                st.warning("Không có runs nào trong thí nghiệm này.")
        else:
            st.warning("Không có thí nghiệm nào được tìm thấy.")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi lấy danh sách thí nghiệm: {e}")
