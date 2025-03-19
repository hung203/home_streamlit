import datetime
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import mlflow
import mlflow.pytorch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import datetime
import random
from turtle import getcanvas
import cv2
from matplotlib import patches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import mlflow
import mlflow.pytorch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from PIL import Image
from streamlit_drawable_canvas import st_canvas
# Tiêu đề ứng dụng
st.title("Phân loại chữ số viết tay MNIST với Self-Training Neural Network")

# Tạo các tab
tab1, tab2, tab3, tab4 = st.tabs(["Lý thuyết", "Huấn luyện", "Dự Đoán", "MLflow"])

# Tab 2: Huấn luyện
with tab2:
    st.header("1. Chọn kích thước và chia tập dữ liệu")

    # Khởi tạo trạng thái dữ liệu
    if "mnist_loaded" not in st.session_state:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        st.session_state.total_samples = mnist.data.shape[0]
        st.session_state.mnist_data = mnist
        st.session_state.mnist_loaded = False
        st.session_state.data_split_done = False

    sample_size = st.number_input(
        "Chọn số lượng mẫu dữ liệu",
        min_value=1000,
        max_value=st.session_state.total_samples,
        value=10000,
        step=1000,
        help="Số lượng mẫu dữ liệu được lấy từ MNIST (tối đa 70,000)."
    )

    test_size = st.slider(
        "Chọn tỷ lệ dữ liệu Test",
        0.1, 0.5, 0.2, 0.05,
        help="Tỷ lệ dữ liệu dùng để kiểm tra mô hình (10%-50%)."
    )
    valid_size = st.slider(
        "Chọn tỷ lệ dữ liệu Validation từ Train",
        0.1, 0.3, 0.2, 0.05,
        help="Tỷ lệ dữ liệu từ tập Train dùng để kiểm tra trong lúc huấn luyện."
    )

    if st.button("Chia tách dữ liệu"):
        mnist = st.session_state.mnist_data
        X, y = mnist.data / 255.0, mnist.target.astype(int)

        if sample_size < st.session_state.total_samples:
            X, _, y, _ = train_test_split(X, y, train_size=sample_size / st.session_state.total_samples, random_state=42, stratify=y)

        st.session_state.X = X
        st.session_state.y = y

        # Chia tập Train và Test
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            st.session_state.X, st.session_state.y, test_size=test_size, random_state=42, stratify=st.session_state.y
        )

        # Chia tập Train thành Train và Validation
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, test_size=valid_size, random_state=42, stratify=y_train_full
        )

        # Từ tập Train, lấy 1% mỗi lớp làm tập labeled ban đầu
        X_labeled = []
        y_labeled = []
        X_unlabeled = []
        y_unlabeled = []
        for digit in range(10):
            digit_indices = np.where(y_train == digit)[0]
            num_samples = len(digit_indices)
            num_labeled = max(1, int(num_samples * 0.01))  # Lấy 1%, đảm bảo ít nhất 1 mẫu
            labeled_indices = np.random.choice(digit_indices, num_labeled, replace=False)
            unlabeled_indices = np.setdiff1d(digit_indices, labeled_indices)

            X_labeled.append(X_train[labeled_indices])
            y_labeled.append(y_train[labeled_indices])
            X_unlabeled.append(X_train[unlabeled_indices])
            y_unlabeled.append(y_train[unlabeled_indices])

        X_labeled = np.concatenate(X_labeled)
        y_labeled = np.concatenate(y_labeled)
        X_unlabeled = np.concatenate(X_unlabeled)
        y_unlabeled = np.concatenate(y_unlabeled)  # Ground truth cho đánh giá

        # Lưu vào session_state
        st.session_state.X_train_labeled = X_labeled
        st.session_state.y_train_labeled = y_labeled
        st.session_state.X_train_unlabeled = X_unlabeled
        st.session_state.y_train_unlabeled = y_unlabeled
        st.session_state.X_valid = X_valid
        st.session_state.y_valid = y_valid
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.data_split_done = True
        st.session_state.mnist_loaded = True

        st.write(f"Dữ liệu đã được chia tách với {sample_size} mẫu!")
        st.write(f"- Dữ liệu Train tổng: {X_train.shape} ({(1 - test_size) * (1 - valid_size) * 100:.1f}%)")
        st.write(f"  + Train có nhãn (1% mỗi lớp): {X_labeled.shape}")
        st.write(f"  + Train không nhãn: {X_unlabeled.shape}")
        st.write(f"- Dữ liệu Validation: {X_valid.shape} ({(1 - test_size) * valid_size * 100:.1f}%)")
        st.write(f"- Dữ liệu Test: {X_test.shape} ({test_size * 100:.1f}%)")

    # Cấu hình huấn luyện Self-Training
    st.header("2. Huấn luyện Neural Network với Self-Training")
    st.subheader("Cấu hình huấn luyện")
    num_epochs = st.number_input("Số epochs mỗi vòng", min_value=1, max_value=50, value=10)
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
    learning_rate = st.number_input("Tốc độ học", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001)
    num_hidden_layers = st.selectbox("Số lớp ẩn", [1, 2, 3], index=0)
    hidden_neurons = st.selectbox("Số nơ-ron mỗi lớp ẩn", [64, 128, 256], index=1)
    activation_function = st.selectbox("Hàm kích hoạt", ["ReLU", "Sigmoid", "Tanh"], index=0)
    threshold = st.slider("Ngưỡng gán Pseudo Label", 0.5, 0.99, 0.95, 0.01)
    max_iterations = st.number_input("Số vòng lặp tối đa", min_value=1, max_value=20, value=5)

    experiment_name = st.text_input(
        "Nhập tên cho thí nghiệm MLflow",
        value=f"Self_Training_MNIST_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    if st.button("Bắt đầu Self-Training"):
        if not st.session_state.get("data_split_done", False):
            st.error("Vui lòng chia tách dữ liệu trước!")
        else:
            X_labeled = st.session_state.X_train_labeled
            y_labeled = st.session_state.y_train_labeled
            X_unlabeled = st.session_state.X_train_unlabeled
            y_unlabeled = st.session_state.y_train_unlabeled  # Ground truth
            X_valid = st.session_state.X_valid
            y_valid = st.session_state.y_valid
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

            # Định nghĩa mô hình Neural Network
            class SimpleNN(nn.Module):
                def __init__(self, num_hidden_layers, hidden_size, activation):
                    super(SimpleNN, self).__init__()
                    layers = [nn.Linear(784, hidden_size)]
                    if activation == "ReLU":
                        layers.append(nn.ReLU())
                    elif activation == "Sigmoid":
                        layers.append(nn.Sigmoid())
                    elif activation == "Tanh":
                        layers.append(nn.Tanh())
                    for _ in range(num_hidden_layers - 1):
                        layers.append(nn.Linear(hidden_size, hidden_size))
                        if activation == "ReLU":
                            layers.append(nn.ReLU())
                        elif activation == "Sigmoid":
                            layers.append(nn.Sigmoid())
                        elif activation == "Tanh":
                            layers.append(nn.Tanh())
                    layers.append(nn.Linear(hidden_size, 10))
                    self.network = nn.Sequential(*layers)

                def forward(self, x):
                    return self.network(x)

            # Thiết lập MLflow
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run() as run:
                # Log các tham số
                mlflow.log_param("num_epochs", num_epochs)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("learning_rate", learning_rate)
                mlflow.log_param("num_hidden_layers", num_hidden_layers)
                mlflow.log_param("hidden_neurons", hidden_neurons)
                mlflow.log_param("activation_function", activation_function)
                mlflow.log_param("threshold", threshold)
                mlflow.log_param("max_iterations", max_iterations)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("valid_size", valid_size)
                mlflow.log_param("sample_size", sample_size)

                progress_bar = st.progress(0)
                status_text = st.empty()
                test_acc_history = []
                valid_acc_history = []

                # Vòng lặp Self-Training
                for iteration in range(max_iterations):
                    # (2) Huấn luyện mô hình trên tập labeled
                    model = SimpleNN(num_hidden_layers, hidden_neurons, activation_function)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                    X_labeled_tensor = torch.tensor(X_labeled, dtype=torch.float32)
                    y_labeled_tensor = torch.tensor(y_labeled, dtype=torch.long)
                    train_dataset = TensorDataset(X_labeled_tensor, y_labeled_tensor)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    model.train()
                    for epoch in range(num_epochs):
                        for inputs, labels in train_loader:
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()

                    # (3) Dự đoán nhãn cho tập unlabeled
                    model.eval()
                    X_unlabeled_tensor = torch.tensor(X_unlabeled, dtype=torch.float32)
                    with torch.no_grad():
                        outputs = model(X_unlabeled_tensor)
                        probabilities = torch.softmax(outputs, dim=1).numpy()
                        predictions = np.argmax(probabilities, axis=1)
                        max_probs = np.max(probabilities, axis=1)

                    # (4) Gán Pseudo Label với ngưỡng
                    pseudo_mask = max_probs >= threshold
                    X_pseudo = X_unlabeled[pseudo_mask]
                    y_pseudo = predictions[pseudo_mask]

                    # (5) Cập nhật tập labeled
                    X_labeled = np.concatenate([X_labeled, X_pseudo])
                    y_labeled = np.concatenate([y_labeled, y_pseudo])
                    X_unlabeled = X_unlabeled[~pseudo_mask]

                    # Đánh giá trên tập Validation
                    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
                    y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
                    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
                    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for inputs, labels in valid_loader:
                            outputs = model(inputs)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    valid_acc = correct / total
                    valid_acc_history.append(valid_acc)

                    # Đánh giá trên tập Test
                    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
                    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            outputs = model(inputs)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    test_acc = correct / total
                    test_acc_history.append(test_acc)

                    # Log kết quả
                    mlflow.log_metric("labeled_size", len(X_labeled), step=iteration)
                    mlflow.log_metric("unlabeled_size", len(X_unlabeled), step=iteration)
                    mlflow.log_metric("valid_accuracy", valid_acc, step=iteration)
                    mlflow.log_metric("test_accuracy", test_acc, step=iteration)

                    progress = (iteration + 1) / max_iterations
                    progress_bar.progress(progress)
                    status_text.text(f"Iteration {iteration+1}/{max_iterations}, Labeled: {len(X_labeled)}, Valid Acc: {valid_acc:.4f}, Test Acc: {test_acc:.4f}")

                    # Dừng nếu không còn dữ liệu unlabeled
                    if len(X_unlabeled) == 0:
                        st.write("Đã gán nhãn hết dữ liệu không nhãn!")
                        break

                # Lưu mô hình
                mlflow.pytorch.log_model(model, "model")
                st.session_state.model = model
                st.session_state.run_id = run.info.run_id

                st.success("Quá trình Self-Training hoàn tất!")

                # Hiển thị biểu đồ tiến trình
                st.subheader("Tiến trình Self-Training")
                fig, ax = plt.subplots()
                ax.plot(range(1, len(test_acc_history) + 1), test_acc_history, label="Test Accuracy")
                ax.plot(range(1, len(valid_acc_history) + 1), valid_acc_history, label="Validation Accuracy")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Accuracy")
                ax.legend()
                st.pyplot(fig)

# Các tab khác (giữ nguyên hoặc điều chỉnh nếu cần)
with tab1:
    st.header("Lý thuyết")
    st.write("Nội dung lý thuyết ở đây...")

with tab3:
    # Hàm tiền xử lý ảnh tải lên
    def preprocess_uploaded_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28))
        image = image.reshape(1, -1) / 255.0
        return image

    # Hàm tiền xử lý ảnh từ canvas
    def preprocess_canvas_image(image_data):
        image = np.array(image_data)[:, :, 0]  # Lấy kênh grayscale
        image = cv2.resize(image, (28, 28))
        image = image.reshape(1, -1) / 255.0
        return image

    # Kiểm tra mô hình đã huấn luyện chưa
    if "model" not in st.session_state:
        st.error("⚠️ Mô hình chưa được huấn luyện! Hãy quay lại tab 'Chia dữ liệu & Huấn luyện' để huấn luyện trước.")
        st.stop()

    st.header("🖍️ Dự đoán chữ số viết tay")
    option = st.radio("🖼️ Chọn phương thức nhập:", ["📂 Tải ảnh lên", "✏️ Vẽ số"])

    if option == "📂 Tải ảnh lên":
        uploaded_file = st.file_uploader("📤 Tải ảnh số viết tay (PNG, JPG)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            processed_image = preprocess_uploaded_image(image)
            st.image(image, caption="📷 Ảnh tải lên", use_column_width=True)

            if st.button("🔮 Dự đoán"):
                model = st.session_state.model
                model.eval()
                with torch.no_grad():
                    input_tensor = torch.tensor(processed_image, dtype=torch.float32)
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1).numpy()[0]
                    prediction = np.argmax(probabilities)
                    st.write(f"🎯 **Dự đoán: {prediction}**")
                    st.write(f"🔢 **Độ tin cậy: {probabilities[prediction] * 100:.2f}%**")

    elif option == "✏️ Vẽ số":
        # Sử dụng st_canvas với các tham số hợp lệ
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",  # Màu tô (trong suốt để không tô nền)
            stroke_width=15,                        # Độ dày nét vẽ
            stroke_color="black",                   # Màu nét vẽ
            background_color="white",               # Màu nền canvas
            width=280,                              # Chiều rộng
            height=280,                             # Chiều cao
            drawing_mode="freedraw",                # Chế độ vẽ tự do
            key="canvas"                            # Khóa duy nhất
        )
        if st.button("🔮 Dự đoán"):
            if canvas_result.image_data is not None:
                processed_canvas = preprocess_canvas_image(canvas_result.image_data)
                model = st.session_state.model
                model.eval()
                with torch.no_grad():
                    input_tensor = torch.tensor(processed_canvas, dtype=torch.float32)
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1).numpy()[0]
                    prediction = np.argmax(probabilities)
                    st.write(f"🎯 **Dự đoán: {prediction}**")
                    st.write(f"🔢 **Độ tin cậy: {probabilities[prediction] * 100:.2f}%**")

# Tab 3: MLflow
with tab4:
    st.header("Tracking MLflow")
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