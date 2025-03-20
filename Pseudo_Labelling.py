import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Đảm bảo không dùng GUI backend
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
import random
import cv2
from matplotlib import patches
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Thiết lập MLflow
mlflow.set_tracking_uri("file:./mlruns")

# Tiêu đề ứng dụng
st.title("Phân loại chữ số viết tay MNIST với Self-Training Neural Network")

# Tạo các tab
tab1, tab2, tab3, tab4 = st.tabs(["Lý thuyết", "Huấn luyện", "Dự Đoán", "MLflow"])

# ------------------ HÀM TẢI DỮ LIỆU (CACHED) ------------------
@st.cache_data
def load_mnist_data(sample_size):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, cache=True)
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    total_samples = mnist.data.shape[0]
    if sample_size < total_samples:
        X, _, y, _ = train_test_split(
            X, y, train_size=sample_size / total_samples, random_state=42, stratify=y
        )
    return X, y, total_samples

# ------------------ HÀM ĐỊNH NGHĨA MÔ HÌNH (CACHED) ------------------
@st.cache_resource
def create_model(num_hidden_layers, hidden_size, activation_function):
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            layers = [nn.Linear(784, hidden_size)]
            if activation_function == "ReLU":
                layers.append(nn.ReLU())
            elif activation_function == "Sigmoid":
                layers.append(nn.Sigmoid())
            elif activation_function == "Tanh":
                layers.append(nn.Tanh())
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                if activation_function == "ReLU":
                    layers.append(nn.ReLU())
                elif activation_function == "Sigmoid":
                    layers.append(nn.Sigmoid())
                elif activation_function == "Tanh":
                    layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_size, 10))
            self.network = nn.Sequential(*layers)
        def forward(self, x):
            return self.network(x)
    return SimpleNN()

# Tab 2: Huấn luyện
with tab2:
    st.header("1. Chọn kích thước và chia tập dữ liệu")

    # Khởi tạo trạng thái dữ liệu
    if "mnist_loaded" not in st.session_state:
        st.session_state.mnist_loaded = False
        st.session_state.data_split_done = False

    sample_size = st.number_input(
        "Chọn số lượng mẫu dữ liệu", min_value=1000, max_value=10000, value=5000, step=1000
    )
    test_size = st.slider("Tỷ lệ dữ liệu Test", 0.1, 0.5, 0.2, 0.05)
    valid_size = st.slider("Tỷ lệ dữ liệu Validation từ Train", 0.1, 0.3, 0.2, 0.05)

    if st.button("Chia tách dữ liệu"):
        X, y, total_samples = load_mnist_data(sample_size)
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.total_samples = total_samples

        # Chia tập Train và Test
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        # Chia tập Train thành Train và Validation
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, test_size=valid_size, random_state=42, stratify=y_train_full
        )

        # Lấy 1% mỗi lớp làm tập labeled ban đầu
        X_labeled, y_labeled, X_unlabeled, y_unlabeled = [], [], [], []
        for digit in range(10):
            digit_indices = np.where(y_train == digit)[0]
            num_samples = len(digit_indices)
            num_labeled = max(1, int(num_samples * 0.01))
            labeled_indices = np.random.choice(digit_indices, num_labeled, replace=False)
            unlabeled_indices = np.setdiff1d(digit_indices, labeled_indices)

            X_labeled.append(X_train[labeled_indices])
            y_labeled.append(y_train[labeled_indices])
            X_unlabeled.append(X_train[unlabeled_indices])
            y_unlabeled.append(y_train[unlabeled_indices])

        st.session_state.X_train_labeled = np.concatenate(X_labeled)
        st.session_state.y_train_labeled = np.concatenate(y_labeled)
        st.session_state.X_train_unlabeled = np.concatenate(X_unlabeled)
        st.session_state.y_train_unlabeled = np.concatenate(y_unlabeled)
        st.session_state.X_valid = X_valid
        st.session_state.y_valid = y_valid
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.data_split_done = True
        st.session_state.mnist_loaded = True

        st.write(f"Dữ liệu đã được chia tách với {sample_size} mẫu!")
        st.write(f"- Train: {X_train.shape}, Labeled: {st.session_state.X_train_labeled.shape}, Unlabeled: {st.session_state.X_train_unlabeled.shape}")
        st.write(f"- Validation: {X_valid.shape}, Test: {X_test.shape}")

    # Cấu hình huấn luyện Self-Training
    st.header("2. Huấn luyện Neural Network với Self-Training")
    st.subheader("Tham số mạng Neural Network")
    num_epochs = st.number_input("Số epochs mỗi vòng", min_value=1, max_value=20, value=5)
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=2)
    learning_rate = st.number_input("Tốc độ học", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001)
    num_hidden_layers = st.number_input("Số lớp ẩn", min_value=1, max_value=3, value=1)
    hidden_neurons = st.selectbox("Số nơ-ron mỗi lớp ẩn", [64, 128], index=0)
    activation_function = st.selectbox("Hàm kích hoạt", ["ReLU", "Sigmoid"], index=0)

    st.subheader("Tham số gán nhãn giả")
    threshold = st.slider("Ngưỡng gán Pseudo Label", 0.5, 0.99, 0.95, 0.01)
    max_iterations = st.number_input("Số vòng lặp tối đa", min_value=1, max_value=10, value=3)

    experiment_name = f"Self_Training_MNIST_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if st.button("Bắt đầu Self-Training"):
        if not st.session_state.get("data_split_done", False):
            st.error("Vui lòng chia tách dữ liệu trước!")
        else:
            X_labeled = st.session_state.X_train_labeled
            y_labeled = st.session_state.y_train_labeled
            X_unlabeled = st.session_state.X_train_unlabeled
            y_unlabeled = st.session_state.y_train_unlabeled
            X_valid = st.session_state.X_valid
            y_valid = st.session_state.y_valid
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

            mlflow.set_experiment(experiment_name)
            with mlflow.start_run() as run:
                # Log các tham số
                params = {
                    "num_epochs": num_epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                    "num_hidden_layers": num_hidden_layers, "hidden_neurons": hidden_neurons,
                    "activation_function": activation_function, "threshold": threshold,
                    "max_iterations": max_iterations, "test_size": test_size, "valid_size": valid_size,
                    "sample_size": sample_size
                }
                mlflow.log_params(params)

                progress_bar = st.progress(0)
                status_text = st.empty()
                test_acc_history = []
                valid_acc_history = []

                # Vòng lặp Self-Training
                for iteration in range(max_iterations):
                    model = create_model(num_hidden_layers, hidden_neurons, activation_function)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                    train_dataset = TensorDataset(
                        torch.tensor(X_labeled, dtype=torch.float32),
                        torch.tensor(y_labeled, dtype=torch.long)
                    )
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    model.train()
                    for epoch in range(num_epochs):
                        for inputs, labels in train_loader:
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()

                    # Dự đoán nhãn giả
                    model.eval()
                    X_unlabeled_tensor = torch.tensor(X_unlabeled, dtype=torch.float32)
                    with torch.no_grad():
                        outputs = model(X_unlabeled_tensor)
                        probabilities = torch.softmax(outputs, dim=1).numpy()
                        predictions = np.argmax(probabilities, axis=1)
                        max_probs = np.max(probabilities, axis=1)

                    # Gán Pseudo Label
                    pseudo_mask = max_probs >= threshold
                    X_pseudo = X_unlabeled[pseudo_mask]
                    y_pseudo = predictions[pseudo_mask]

                    # Cập nhật tập labeled
                    X_labeled = np.concatenate([X_labeled, X_pseudo])
                    y_labeled = np.concatenate([y_labeled, y_pseudo])
                    X_unlabeled = X_unlabeled[~pseudo_mask]

                    # Đánh giá Validation
                    valid_dataset = TensorDataset(
                        torch.tensor(X_valid, dtype=torch.float32),
                        torch.tensor(y_valid, dtype=torch.long)
                    )
                    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
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

                    # Đánh giá Test
                    test_dataset = TensorDataset(
                        torch.tensor(X_test, dtype=torch.float32),
                        torch.tensor(y_test, dtype=torch.long)
                    )
                    test_loader = DataLoader(test_dataset, batch_size=batch_size)
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

                    # Log MLflow
                    mlflow.log_metrics({
                        "labeled_size": len(X_labeled),
                        "unlabeled_size": len(X_unlabeled),
                        "valid_accuracy": valid_acc,
                        "test_accuracy": test_acc
                    }, step=iteration)

                    progress_bar.progress((iteration + 1) / max_iterations)
                    status_text.text(f"Iteration {iteration+1}: Valid Acc: {valid_acc:.4f}, Test Acc: {test_acc:.4f}")

                    if len(X_unlabeled) == 0:
                        break

                # Lưu mô hình
                mlflow.pytorch.log_model(model, "model")
                st.session_state.model = model
                st.session_state.run_id = run.info.run_id

                st.success("Self-Training hoàn tất!")
                st.write(f"Valid Acc: {valid_acc_history[-1]:.4f}, Test Acc: {test_acc_history[-1]:.4f}")

                # Biểu đồ
                fig, ax = plt.subplots()
                ax.plot(range(1, len(test_acc_history) + 1), test_acc_history, label="Test Acc")
                ax.plot(range(1, len(valid_acc_history) + 1), valid_acc_history, label="Valid Acc")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Accuracy")
                ax.legend()
                st.pyplot(fig)

                # Hiển thị 5 mẫu ví dụ (giảm từ 10 để nhẹ hơn)
                st.subheader("5 mẫu ví dụ từ tập Test")
                random_indices = np.random.choice(len(X_test), 5, replace=False)
                X_samples = X_test[random_indices]
                y_true = y_test[random_indices]
                with torch.no_grad():
                    outputs = model(torch.tensor(X_samples, dtype=torch.float32))
                    y_pred = torch.argmax(outputs, dim=1).numpy()

                fig, axes = plt.subplots(1, 5, figsize=(10, 2))
                for i, (sample, true, pred) in enumerate(zip(X_samples, y_true, y_pred)):
                    axes[i].imshow(sample.reshape(28, 28), cmap='gray')
                    axes[i].set_title(f"T: {true}\nP: {pred}")
                    axes[i].axis('off')
                plt.tight_layout()
                st.pyplot(fig)

# Tab 1: Lý thuyết (giữ nguyên, lược giản văn bản)
with tab1:
    st.header(":book: 1. Pseudo-Labeling là gì?")
    st.write("Pseudo-Labeling tận dụng dữ liệu không nhãn bằng cách dự đoán nhãn giả và huấn luyện lại mô hình.")

    st.header(":question: 2. Tại sao cần?")
    st.write("- Ít dữ liệu có nhãn.\n- Tăng độ chính xác.\n- Ứng dụng thực tế (như MNIST).")

    st.header(":gear: 3. Quy trình Self-Training")
    st.write("1. \( L = \{(x_i, y_i)\}_{i=1}^{N_L} \), \( U = \{x_j\}_{j=1}^{N_U} \)")
    st.write("2. \( \min_{\theta} \sum \text{Loss}(f(x_i; \theta), y_i) \)")
    st.write("3. \( y_{pseudo,j} = \arg\max_{k} (p_j(k)) \)")
    st.write("4. \( \max_{k} (p_j(k)) \geq \tau \)")
    st.write("5. \( L = L \cup \{(x_j, y_{pseudo,j})\} \), \( U = U \setminus \{x_j\} \)")
    st.write("6. Lặp lại đến khi \( U = \emptyset \).")

    st.header("4. Ưu điểm & Hạn chế")
    st.write("Ưu: Đơn giản, hiệu quả.\nHạn: Nhạy với nhiễu, phụ thuộc ngưỡng.")

    st.header(":tada: 5. Kết luận")
    st.write("Pseudo-Labeling mạnh mẽ khi ít dữ liệu có nhãn.")

# Tab 3: Dự đoán
with tab3:
    def preprocess_uploaded_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28))
        return image.reshape(1, -1) / 255.0

    def preprocess_canvas_image(image_data):
        image = np.array(image_data)[:, :, 0]
        image = cv2.resize(image, (28, 28))
        return image.reshape(1, -1) / 255.0

    if "model" not in st.session_state:
        st.error("Vui lòng huấn luyện mô hình trước!")
    else:
        st.header("Dự đoán chữ số viết tay")
        option = st.radio("Chọn phương thức:", ["Tải ảnh lên", "Vẽ số"])

        if option == "Tải ảnh lên":
            uploaded_file = st.file_uploader("Tải ảnh (PNG, JPG)", type=["png", "jpg", "jpeg"])
            if uploaded_file:
                image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
                processed_image = preprocess_uploaded_image(image)
                st.image(image, caption="Ảnh tải lên", width=100)

                if st.button("Dự đoán"):
                    model = st.session_state.model
                    model.eval()
                    with torch.no_grad():
                        outputs = model(torch.tensor(processed_image, dtype=torch.float32))
                        probabilities = torch.softmax(outputs, dim=1).numpy()[0]
                        prediction = np.argmax(probabilities)
                        st.write(f"Dự đoán: {prediction}, Độ tin cậy: {probabilities[prediction] * 100:.2f}%")

        elif option == "Vẽ số":
            canvas_result = st_canvas(
                stroke_width=15, stroke_color="white", background_color="black",
                width=280, height=280, drawing_mode="freedraw", key="canvas"
            )
            if st.button("Dự đoán") and canvas_result.image_data is not None:
                processed_canvas = preprocess_canvas_image(canvas_result.image_data)
                model = st.session_state.model
                model.eval()
                with torch.no_grad():
                    outputs = model(torch.tensor(processed_canvas, dtype=torch.float32))
                    probabilities = torch.softmax(outputs, dim=1).numpy()[0]
                    prediction = np.argmax(probabilities)
                    st.write(f"Dự đoán: {prediction}, Độ tin cậy: {probabilities[prediction] * 100:.2f}%")

# Tab 4: MLflow
with tab4:
    st.header("Tracking MLflow")
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        experiments = mlflow.search_experiments()

        if experiments:
            experiment_data = [{"ID": exp.experiment_id, "Name": exp.name} for exp in experiments]
            st.dataframe(pd.DataFrame(experiment_data))

            selected_exp_name = st.selectbox("Chọn thí nghiệm", [exp.name for exp in experiments])
            selected_exp_id = next(exp.experiment_id for exp in experiments if exp.name == selected_exp_name)

            runs = mlflow.search_runs(selected_exp_id)
            if not runs.empty:
                st.dataframe(runs[["run_id", "start_time", "status"]])
                selected_run_id = st.selectbox("Chọn run", runs["run_id"])
                run = mlflow.get_run(selected_run_id)
                st.write(f"Run ID: {run.info.run_id}")
                st.write(f"Metrics: {run.data.metrics}")
                st.write(f"Params: {run.data.params}")
        else:
            st.warning("Không có thí nghiệm nào.")
    except Exception as e:
        st.error(f"Lỗi: {e}")
