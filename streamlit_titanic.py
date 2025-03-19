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

# Tạo các tab
tab1, tab2, tab3, tab4 = st.tabs(["Xử lý dữ liệu", "Huấn luyện", "Dự đoán", "MLflow"])

# --------------------- Tab 1: Xử lý dữ liệu ---------------------
with tab1:
    st.header("1. Tải dữ liệu")

    # Cache dữ liệu mặc định
    @st.cache_data
    def load_default_data():
        return pd.read_csv("titanic.csv")

    # Fragment cho tải dữ liệu
    @st.fragment
    def load_data_interface():
        if "df" not in st.session_state:
            uploaded_file = st.file_uploader("Tải lên file CSV", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
            else:
                df = load_default_data()
            st.session_state.df = df.copy()
        st.write("Dữ liệu ban đầu:")
        st.write(st.session_state.df)

    load_data_interface()

    st.header("2. Xử lý giá trị thiếu")
    df = st.session_state.df
    missing_data = df.isnull().sum().reset_index()
    missing_data.columns = ['Column', 'Missing Count']
    st.write("Bảng số lượng giá trị thiếu:")
    st.write(missing_data)

    missing_cols = df.columns[df.isnull().any()].tolist()
    user_missing_choices = {}
    st.markdown("### Chọn phương pháp xử lý cho các cột có giá trị thiếu:")
    for col in missing_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            user_missing_choices[col] = st.selectbox(
                f"Phương pháp xử lý cho '{col}'", 
                options=["Giá trị trung bình", "Giá trị trung vị", "Giá trị xuất hiện nhiều nhất"],
                key=f"method_{col}"
            )
        else:
            user_missing_choices[col] = st.selectbox(
                f"Phương pháp xử lý cho '{col}'", 
                options=["Giá trị xuất hiện nhiều nhất"],
                key=f"method_{col}"
            )

    # Fragment cho xử lý giá trị thiếu
    @st.fragment
    def process_missing_values():
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
                    df[col].fillna(df[col].mode()[0], inplace=True)
            st.session_state.df = df.copy()
            st.session_state.missing_processed = True
            st.success("Xử lý giá trị thiếu thành công!")
            st.write("Dữ liệu sau khi xử lý giá trị thiếu:")
            st.write(st.session_state.df)

    process_missing_values()

    st.header("3. Mã hóa dữ liệu")
    # Fragment cho mã hóa dữ liệu
    @st.fragment
    def encode_data():
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
            st.write("Dữ liệu đã được mã hóa trước đó:")
            st.write(st.session_state.df)

    encode_data()

    st.header("4. Xóa các cột không cần thiết")
    default_cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    selected_cols_to_drop = st.multiselect(
        "Chọn các cột muốn xóa:", 
        options=df.columns.tolist(), 
        default=[col for col in default_cols_to_drop if col in df.columns]
    )

    # Fragment cho xóa cột
    @st.fragment
    def drop_columns():
        if st.button("Xóa các cột đã chọn"):
            if selected_cols_to_drop:
                df.drop(selected_cols_to_drop, axis=1, inplace=True)
                st.session_state.df = df.copy()
                st.success("Đã xóa các cột: " + ", ".join(selected_cols_to_drop))
            else:
                st.info("Không có cột nào được chọn.")
            st.write("Dữ liệu sau khi xóa các cột không cần thiết:")
            st.write(st.session_state.df)

    drop_columns()

    st.header("5. Chuẩn hóa dữ liệu")
    # Fragment cho chuẩn hóa dữ liệu
    @st.fragment
    def standardize_data():
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

    standardize_data()

# --------------------- Tab 2: Huấn luyện ---------------------
with tab2:
    st.header("Chia dữ liệu")
    df = st.session_state.df

    if "Survived" not in df.columns:
        st.error("Cột 'Survived' không tồn tại trong dữ liệu. Vui lòng không xóa cột này để tiến hành chia dữ liệu và huấn luyện mô hình.")
    else:
        X = df.drop("Survived", axis=1)
        y = df["Survived"]

        # Fragment cho chia dữ liệu
        @st.fragment
        def split_data():
            test_size = st.slider("Chọn tỉ lệ tập Test (%)", min_value=10, max_value=50, value=20, step=5) / 100.0
            valid_size = st.slider("Chọn tỉ lệ tập Valid (%) trên tập Train", min_value=10, max_value=50, value=20, step=5) / 100.0
            if st.button("Chia dữ liệu"):
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

        split_data()

    st.header("Huấn luyện & Kiểm thử mô hình")
    if "data_split_done" in st.session_state and st.session_state.data_split_done:
        algorithm = st.selectbox("Thuật toán:", ["Multiple Regression", "Polynomial Regression"])
        
        st.markdown("### Tùy chọn thông số của mô hình")
        if algorithm == "Multiple Regression":
            model = LinearRegression(fit_intercept=True)
        else:
            degree = st.number_input("Chọn bậc của đa thức:", min_value=2, max_value=5, value=2)
            poly_features = PolynomialFeatures(degree=degree, include_bias=True)
            linear_model = LinearRegression(fit_intercept=True)
            model = Pipeline([('poly', poly_features), ('linear', linear_model)])

        experiment_name = st.text_input(
            "Nhập tên cho thí nghiệm MLflow", 
            value=f"{algorithm}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        # Fragment cho huấn luyện mô hình
        @st.fragment
        def train_model():
            if st.button("Huấn luyện mô hình"):
                mlflow.set_experiment(experiment_name)
                st.session_state.experiment_name = experiment_name
                st.write("Tên thí nghiệm:", experiment_name)

                with mlflow.start_run() as run:
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)
                    try:
                        scores = cross_val_score(model, st.session_state.X_train, st.session_state.y_train, 
                                                cv=cv, scoring='r2')
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

                        st.markdown("### Tham số của mô hình")
                        if algorithm == "Multiple Regression":
                            coef_df = pd.DataFrame({
                                "Feature": st.session_state.X_train.columns,
                                "Coefficient": model.coef_
                            })
                            st.write(coef_df)
                            st.write("Hệ số chặn (Intercept):", model.intercept_)
                        else:
                            feature_names = model.named_steps['poly'].get_feature_names_out(st.session_state.X_train.columns)
                            coef_df = pd.DataFrame({
                                "Feature": feature_names,
                                "Coefficient": model.named_steps['linear'].coef_
                            })
                            st.write(coef_df)
                            st.write("Hệ số chặn của mô hình tuyến tính:", model.named_steps['linear'].intercept_)

                        st.session_state.trained_model = model
                        mlflow.sklearn.log_model(model, "model")
                    except Exception as e:
                        st.error("Lỗi khi huấn luyện mô hình hoặc dự đoán: " + str(e))

        train_model()
    else:
        st.info("Chưa có dữ liệu được chia, vui lòng thực hiện bước chia dữ liệu.")

# --------------------- Tab 3: Dự đoán ---------------------
with tab3:
    st.header("Dự đoán sự sống sót")
    # Fragment cho dự đoán
    @st.fragment
    def predict_survival():
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
                if "trained_model" not in st.session_state:
                    st.error("Mô hình chưa được huấn luyện. Vui lòng huấn luyện mô hình trước.")
                else:
                    sex_encoded = 1 if sex == "female" else 0
                    embarked_encoded = {"C": 0, "Q": 1, "S": 2}[embarked]
                    input_df = pd.DataFrame({
                        "Pclass": [pclass], "Sex": [sex_encoded], "Age": [age],
                        "SibSp": [sibsp], "Parch": [parch], "Fare": [fare],
                        "Embarked": [embarked_encoded]
                    })
                    if "scaler" in st.session_state:
                        scaler = st.session_state.scaler
                        cols_to_scale = [col for col in ["Age", "Fare", "SibSp", "Parch", "Pclass", "Embarked"] if col in input_df.columns]
                        input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
                    else:
                        st.warning("Chưa có scaler được lưu. Kết quả dự đoán có thể không chính xác nếu dữ liệu chưa chuẩn hóa.")

                    try:
                        model = st.session_state.trained_model
                        prediction = model.predict(input_df)
                        predicted_class = 1 if prediction[0] >= 0.5 else 0
                        result = "🌟 Survived!" if predicted_class == 1 else "💀 Did Not Survive"
                        st.subheader(f"Prediction Result: {result}")
                    except Exception as e:
                        st.error(f"Lỗi khi dự đoán: {str(e)}")

    predict_survival()

# --------------------- Tab 4: MLflow ---------------------
with tab4:
    st.header("Tracking MLflow")
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        experiments = mlflow.search_experiments()

        if experiments:
            st.write("#### Danh sách thí nghiệm")
            experiment_data = [{"Experiment ID": exp.experiment_id, "Experiment Name": exp.name, "Artifact Location": exp.artifact_location} for exp in experiments]
            df_experiments = pd.DataFrame(experiment_data)
            st.dataframe(df_experiments)

            selected_exp_name = st.selectbox("🔍 Chọn thí nghiệm để xem chi tiết", options=[exp.name for exp in experiments])
            selected_exp_id = next(exp.experiment_id for exp in experiments if exp.name == selected_exp_name)
            runs = mlflow.search_runs(selected_exp_id)

            if not runs.empty:
                st.write("#### Danh sách runs")
                st.dataframe(runs)
                selected_run_id = st.selectbox("🔍 Chọn run để xem chi tiết", options=runs["run_id"])
                run = mlflow.get_run(selected_run_id)

                st.write("##### Thông tin run")
                st.write(f"*Run ID:* {run.info.run_id}")
                st.write(f"*Experiment ID:* {run.info.experiment_id}")
                st.write(f"*Start Time:* {run.info.start_time}")
                st.write("##### Metrics")
                st.json(run.data.metrics)
                st.write("##### Params")
                st.json(run.data.params)

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
