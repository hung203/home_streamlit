import streamlit as st
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
# Tiêu đề ứng dụng
st.title("Phân loại chữ số viết tay MNIST với Streamlit và MLflow")

tab1, tab2, tab3, tab4 = st.tabs([
    "Xử lý dữ liệu",
    "Huấn luyện",
    "Dự đoán",
    'Mlflow'
])

# ------------------------
# Bước 1: Xử lý dữ liệu
# ------------------------
with tab1:
    st.header("1. Xử lý dữ liệu")

    # Kiểm tra nếu dữ liệu đã được tải chưa
    if "mnist_loaded" not in st.session_state:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        st.session_state.total_samples = mnist.data.shape[0]  # Tổng số mẫu
        st.session_state.mnist_data = mnist  # Lưu dữ liệu gốc
        st.session_state.mnist_loaded = False  # Chưa tải mẫu cụ thể

    # Chọn số lượng mẫu sử dụng
    sample_size = st.number_input(
        "Chọn số lượng mẫu dữ liệu sử dụng", 
        min_value=1000, 
        max_value=st.session_state.total_samples, 
        value=st.session_state.total_samples, 
        step=1000
    )

    if st.button("Tải dữ liệu MNIST"):
        mnist = st.session_state.mnist_data
        X, y = mnist.data / 255.0, mnist.target.astype(int)
        
        # Chọn số lượng mẫu dữ liệu theo yêu cầu
        if sample_size < st.session_state.total_samples:
            X, _, y, _ = train_test_split(X, y, train_size=sample_size, random_state=42, stratify=y)
        
        # Lưu dữ liệu vào session_state
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.mnist_loaded = True
        st.session_state.selected_sample_size = sample_size
        st.write(f"Dữ liệu MNIST đã được tải với {sample_size} mẫu!")

    # Chỉ hiển thị bước chia tách khi dữ liệu đã tải
    if st.session_state.mnist_loaded:
        st.header("2. Chia tách dữ liệu")

        test_size = st.slider("Chọn tỷ lệ dữ liệu Test", 0.1, 0.5, 0.2, 0.05)
        valid_size = st.slider("Chọn tỷ lệ dữ liệu Validation từ Train", 0.1, 0.3, 0.2, 0.05)

        if st.button("Chia tách dữ liệu"):
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                st.session_state.X, st.session_state.y, test_size=test_size, random_state=42, stratify=st.session_state.y
            )
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train_full, y_train_full, test_size=valid_size, random_state=42, stratify=y_train_full
            )

            # Lưu vào session_state
            st.session_state.X_train = X_train
            st.session_state.X_valid = X_valid
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_valid = y_valid
            st.session_state.y_test = y_test
            st.session_state.data_split_done = True

        # Hiển thị thông tin sau khi chia tách
        if st.session_state.get("data_split_done", False):
            st.write(f"Dữ liệu Train: {st.session_state.X_train.shape}")
            st.write(f"Dữ liệu Validation: {st.session_state.X_valid.shape}")
            st.write(f"Dữ liệu Test: {st.session_state.X_test.shape}")
    
    # Hiển thị hình ảnh minh họa
    st.subheader("Ví dụ một vài hình ảnh minh họa")
    
    if st.session_state.mnist_loaded:
        X = st.session_state.X
        y = st.session_state.y

        # Nút làm mới hình ảnh
        if st.button("🔄 Hiển thị ảnh mới"):
            st.session_state.example_images = random.sample(range(len(X)), 5)

        indices = st.session_state.get("example_images", random.sample(range(len(X)), 5))

        fig, axs = plt.subplots(1, 5, figsize=(12, 3))
        for i, idx in enumerate(indices):
            img = X[idx].reshape(28, 28)
            axs[i].imshow(img, cmap='gray')
            axs[i].axis('off')
            axs[i].set_title(f"Label: {y[idx]}")
        
        st.pyplot(fig)
    else:
        st.warning("Vui lòng tải dữ liệu trước khi hiển thị hình ảnh!")
# ------------------------
# Bước 3: Huấn luyện và đánh giá mô hình
# ------------------------
with tab2:
    st.header("3. Huấn luyện và đánh giá mô hình")

    # Chọn mô hình
    model_choice = st.selectbox("Chọn mô hình", ["Decision Tree", "SVM"], key="model_choice")
    
    if model_choice == "Decision Tree":
        max_depth = st.slider("Chọn độ sâu tối đa của cây quyết định", 1, 50, 20)
        min_samples_split = st.slider("Số mẫu tối thiểu để chia một node", 2, 20, 2)
        min_samples_leaf = st.slider("Số mẫu tối thiểu trong một lá", 1, 10, 1)
    
    elif model_choice == "SVM":
        kernel = st.selectbox("Chọn kernel", ["linear", "poly", "rbf", "sigmoid"], index=2)
        
        if kernel == "linear":
            C = st.number_input("Chọn giá trị C", min_value=0.01, max_value=100.0, value=10.0, step=0.1)
            gamma = None  # Không sử dụng gamma với kernel linear
        else:
            gamma = st.number_input("Chọn giá trị gamma", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001)
            C = None  # Không sử dụng C với kernel rbf, poly, sigmoid
        
        degree = st.slider("Bậc của kernel (chỉ dùng cho poly)", 2, 5, 3) if kernel == "poly" else None
    
    # Nút huấn luyện
    if st.button("Huấn luyện mô hình"):
        if "X_train" not in st.session_state or "X_valid" not in st.session_state:
            st.error("Bạn cần chia tách dữ liệu trước!")
            st.stop()
        
        X_train_used = st.session_state.X_train
        y_train_used = st.session_state.y_train
        X_valid = st.session_state.X_valid
        y_valid = st.session_state.y_valid

        # Tạo tên thí nghiệm tự động dựa trên tên mô hình và thời gian hiện tại
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"Experiment_{model_choice}_{timestamp}"
        # Thiết lập tên thí nghiệm cho mlflow (nếu thí nghiệm chưa tồn tại, mlflow sẽ tạo mới)
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            # Log tên thí nghiệm dưới dạng tham số
            mlflow.log_param("experiment_name", experiment_name)
            
            if model_choice == "Decision Tree":
                model = DecisionTreeClassifier(
                    max_depth=max_depth, 
                    min_samples_split=min_samples_split, 
                    min_samples_leaf=min_samples_leaf, 
                    random_state=42
                )
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("min_samples_split", min_samples_split)
                mlflow.log_param("min_samples_leaf", min_samples_leaf)
    
            else:  # SVM
                scaler = StandardScaler()
                X_train_used_scaled = scaler.fit_transform(X_train_used)
                X_valid_scaled = scaler.transform(X_valid)
                st.session_state.scaler = scaler
                
                model_params = {"kernel": kernel, "random_state": 42}
                
                if kernel == "linear":
                    model_params["C"] = C
                    mlflow.log_param("C", C)
                else:
                    model_params["gamma"] = gamma
                    mlflow.log_param("gamma", gamma)
    
                if kernel == "poly":
                    model_params["degree"] = degree
                    mlflow.log_param("degree", degree)
                
                model = SVC(**model_params)
                
                X_train_used, X_valid = X_train_used_scaled, X_valid_scaled
    
            # Huấn luyện mô hình
            model.fit(X_train_used, y_train_used)
            y_pred = model.predict(X_valid)
    
            # Lưu các kết quả và mô hình vào session_state
            st.session_state.model = model
            st.session_state.trained_model_name = model_choice
            st.session_state.train_accuracy = accuracy_score(y_valid, y_pred)
            st.session_state.train_report = classification_report(y_valid, y_pred)
            
            mlflow.log_param("model", model_choice)
            mlflow.log_metric("accuracy", st.session_state.train_accuracy)
            mlflow.sklearn.log_model(model, "model")
    
        # Lưu tên thí nghiệm vào session_state và hiển thị ra giao diện
        st.session_state.experiment_name = experiment_name
        st.write("Tên thí nghiệm:", experiment_name)
    
    # Hiển thị kết quả sau khi huấn luyện
    if "train_accuracy" in st.session_state:
        st.write(f"🔹 **Độ chính xác trên tập validation:** {st.session_state.train_accuracy:.4f}")
    if "train_report" in st.session_state:
        st.text("Báo cáo phân loại:")
        st.text(st.session_state.train_report)


# ------------------------
# Bước 4: Demo dự đoán
# ------------------------

with tab3:
    # Hàm tiền xử lý ảnh tải lên
    def preprocess_uploaded_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển thành ảnh xám
        image = cv2.resize(image, (28, 28))  # Resize về 28x28
        image = image.reshape(1, -1) / 255.0  # Chuẩn hóa về [0,1]
        return image

    # Hàm tiền xử lý ảnh từ canvas
    def preprocess_canvas_image(image_data):
        image = np.array(image_data)[:, :, 0]  # Lấy kênh grayscale
        image = cv2.resize(image, (28, 28))  # Resize về 28x28
        image = image.reshape(1, -1) / 255.0  # Chuẩn hóa
        return image

    # Kiểm tra mô hình đã huấn luyện chưa
    if "model" not in st.session_state:
        st.error("⚠️ Mô hình chưa được huấn luyện! Hãy quay lại tab trước để huấn luyện trước khi dự đoán.")
        st.stop()

    # Chọn phương thức nhập ảnh
    st.header("🖍️ Dự đoán chữ số viết tay")
    option = st.radio("🖼️ Chọn phương thức nhập:", ["📂 Tải ảnh lên", "✏️ Vẽ số"])

    # 📂 Xử lý ảnh tải lên
    if option == "📂 Tải ảnh lên":
        uploaded_file = st.file_uploader("📤 Tải ảnh số viết tay (PNG, JPG)", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            processed_image = preprocess_uploaded_image(image)

            # Hiển thị ảnh
            st.image(image, caption="📷 Ảnh tải lên", use_column_width=True)

            # Dự đoán số
            if st.button("🔮 Dự đoán"):
                model = st.session_state.model

                if st.session_state.trained_model_name == "SVM" and "scaler" in st.session_state:
                    processed_image = st.session_state.scaler.transform(processed_image)

                prediction = model.predict(processed_image)[0]
                probabilities = model.predict_proba(processed_image)[0]

                st.write(f"🎯 **Dự đoán: {prediction}**")
                st.write(f"🔢 **Độ tin cậy: {probabilities[prediction] * 100:.2f}%**")

    # ✏️ Vẽ số trên canvas
    elif option == "✏️ Vẽ số":
        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=15,
            stroke_color="black",
            background_color="white",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas"
        )

        if st.button("🔮 Dự đoán"):
            if canvas_result.image_data is not None:
                processed_canvas = preprocess_canvas_image(canvas_result.image_data)

                model = st.session_state.model

                if st.session_state.trained_model_name == "SVM" and "scaler" in st.session_state:
                    processed_canvas = st.session_state.scaler.transform(processed_canvas)

                prediction = model.predict(processed_canvas)[0]
                probabilities = model.predict_proba(processed_canvas)[0]

                st.write(f"🎯 **Dự đoán: {prediction}**")
                st.write(f"🔢 **Độ tin cậy: {probabilities[prediction] * 100:.2f}%**")

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
