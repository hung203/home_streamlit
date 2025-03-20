import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend không GUI cho Streamlit Cloud
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
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Tiêu đề ứng dụng
st.title("Phân loại chữ số viết tay MNIST với Self-Training Neural Network")

# Tạo các tab
tab1, tab2, tab3, tab4 = st.tabs(["Lý thuyết", "Huấn luyện", "Dự Đoán", "MLflow"])

# Hàm tải dữ liệu MNIST với cache
@st.cache_data
def load_mnist_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    return mnist

# Hàm chia dữ liệu
def split_data(mnist, sample_size, test_size, valid_size):
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    if sample_size < mnist.data.shape[0]:
        X, _, y, _ = train_test_split(X, y, train_size=sample_size / mnist.data.shape[0], 
                                     random_state=42, stratify=y)
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=valid_size, random_state=42, stratify=y_train_full)
    
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
    
    return (np.concatenate(X_labeled), np.concatenate(y_labeled),
            np.concatenate(X_unlabeled), np.concatenate(y_unlabeled),
            X_valid, y_valid, X_test, y_test)

# Định nghĩa mô hình Neural Network
class SimpleNN(nn.Module):
    def __init__(self, num_hidden_layers, hidden_size, activation):
        super(SimpleNN, self).__init__()
        layers = [nn.Linear(784, hidden_size)]
        activation_fn = {"ReLU": nn.ReLU(), "Sigmoid": nn.Sigmoid(), "Tanh": nn.Tanh()}[activation]
        layers.append(activation_fn)
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation_fn)
        layers.append(nn.Linear(hidden_size, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Tab 2: Huấn luyện
with tab2:
    st.header("1. Chọn kích thước và chia tập dữ liệu")
    
    if "mnist_loaded" not in st.session_state:
        st.session_state.mnist_loaded = False
        st.session_state.data_split_done = False
    
    mnist = load_mnist_data()
    total_samples = mnist.data.shape[0]
    
    sample_size = st.number_input("Chọn số lượng mẫu dữ liệu", min_value=1000, 
                                  max_value=total_samples, value=10000, step=1000)
    test_size = st.slider("Tỷ lệ dữ liệu Test", 0.1, 0.5, 0.2, 0.05)
    valid_size = st.slider("Tỷ lệ dữ liệu Validation từ Train", 0.1, 0.3, 0.2, 0.05)

    if st.button("Chia tách dữ liệu"):
        with st.spinner("Đang chia tách dữ liệu..."):
            (X_labeled, y_labeled, X_unlabeled, y_unlabeled, 
             X_valid, y_valid, X_test, y_test) = split_data(mnist, sample_size, test_size, valid_size)
            
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
            st.write(f"- Train có nhãn: {X_labeled.shape}")
            st.write(f"- Train không nhãn: {X_unlabeled.shape}")
            st.write(f"- Validation: {X_valid.shape}")
            st.write(f"- Test: {X_test.shape}")

    st.header("2. Huấn luyện Neural Network với Self-Training")
    st.subheader("Tham số Neural Network")
    num_epochs = st.number_input("Số epochs mỗi vòng", min_value=1, max_value=50, value=10)
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
    learning_rate = st.number_input("Tốc độ học", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001)
    num_hidden_layers = st.number_input("Số lớp ẩn", min_value=1, max_value=5, value=1)
    hidden_neurons = st.selectbox("Số nơ-ron mỗi lớp ẩn", [32, 64, 128], index=1)
    activation_function = st.selectbox("Hàm kích hoạt", ["ReLU", "Sigmoid", "Tanh"], index=0)

    st.subheader("Tham số Pseudo Labeling")
    threshold = st.slider("Ngưỡng gán Pseudo Label", 0.5, 0.99, 0.95, 0.01)
    max_iterations = st.number_input("Số vòng lặp tối đa", min_value=1, max_value=10, value=5)

    experiment_name = st.text_input("Tên thí nghiệm MLflow", 
                                    value=f"Self_Training_MNIST_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

    if st.button("Bắt đầu Self-Training"):
        if not st.session_state.get("data_split_done", False):
            st.error("Vui lòng chia tách dữ liệu trước!")
        else:
            with st.spinner("Đang huấn luyện mô hình..."):
                try:
                    X_labeled = st.session_state.X_train_labeled
                    y_labeled = st.session_state.y_train_labeled
                    X_unlabeled = st.session_state.X_train_unlabeled
                    X_valid = st.session_state.X_valid
                    y_valid = st.session_state.y_valid
                    X_test = st.session_state.X_test
                    y_test = st.session_state.y_test

                    mlflow.set_experiment(experiment_name)
                    with mlflow.start_run() as run:
                        mlflow.log_params({
                            "num_epochs": num_epochs, "batch_size": batch_size, 
                            "learning_rate": learning_rate, "num_hidden_layers": num_hidden_layers,
                            "hidden_neurons": hidden_neurons, "activation_function": activation_function,
                            "threshold": threshold, "max_iterations": max_iterations,
                            "test_size": test_size, "valid_size": valid_size, "sample_size": sample_size
                        })

                        progress_bar = st.progress(0)
                        test_acc_history, valid_acc_history = [], []

                        for iteration in range(max_iterations):
                            model = SimpleNN(num_hidden_layers, hidden_neurons, activation_function)
                            criterion = nn.CrossEntropyLoss()
                            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                            train_dataset = TensorDataset(torch.tensor(X_labeled, dtype=torch.float32),
                                                        torch.tensor(y_labeled, dtype=torch.long))
                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                            model.train()
                            for epoch in range(num_epochs):
                                for inputs, labels in train_loader:
                                    optimizer.zero_grad()
                                    outputs = model(inputs)
                                    loss = criterion(outputs, labels)
                                    loss.backward()
                                    optimizer.step()

                            model.eval()
                            with torch.no_grad():
                                X_unlabeled_tensor = torch.tensor(X_unlabeled, dtype=torch.float32)
                                outputs = model(X_unlabeled_tensor)
                                probs = torch.softmax(outputs, dim=1).numpy()
                                predictions = np.argmax(probs, axis=1)
                                max_probs = np.max(probs, axis=1)

                            pseudo_mask = max_probs >= threshold
                            X_labeled = np.concatenate([X_labeled, X_unlabeled[pseudo_mask]])
                            y_labeled = np.concatenate([y_labeled, predictions[pseudo_mask]])
                            X_unlabeled = X_unlabeled[~pseudo_mask]

                            valid_loader = DataLoader(TensorDataset(torch.tensor(X_valid, dtype=torch.float32),
                                                                   torch.tensor(y_valid, dtype=torch.long)),
                                                     batch_size=batch_size, shuffle=False)
                            correct, total = 0, 0
                            with torch.no_grad():
                                for inputs, labels in valid_loader:
                                    outputs = model(inputs)
                                    _, predicted = torch.max(outputs.data, 1)
                                    total += labels.size(0)
                                    correct += (predicted == labels).sum().item()
                            valid_acc = correct / total
                            valid_acc_history.append(valid_acc)

                            test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                                                  torch.tensor(y_test, dtype=torch.long)),
                                                    batch_size=batch_size, shuffle=False)
                            correct, total = 0, 0
                            with torch.no_grad():
                                for inputs, labels in test_loader:
                                    outputs = model(inputs)
                                    _, predicted = torch.max(outputs.data, 1)
                                    total += labels.size(0)
                                    correct += (predicted == labels).sum().item()
                            test_acc = correct / total
                            test_acc_history.append(test_acc)

                            mlflow.log_metrics({"labeled_size": len(X_labeled), "unlabeled_size": len(X_unlabeled),
                                               "valid_accuracy": valid_acc, "test_accuracy": test_acc}, step=iteration)

                            progress_bar.progress((iteration + 1) / max_iterations)
                            st.write(f"Iteration {iteration+1}/{max_iterations}, Valid Acc: {valid_acc:.4f}, Test Acc: {test_acc:.4f}")

                            if len(X_unlabeled) == 0:
                                st.write("Đã gán nhãn hết dữ liệu không nhãn!")
                                break

                        mlflow.pytorch.log_model(model, "model")
                        st.session_state.model = model
                        st.session_state.run_id = run.info.run_id

                        st.success("Self-Training hoàn tất!")
                        st.write(f"Độ chính xác Validation: {valid_acc_history[-1]:.4f}")
                        st.write(f"Độ chính xác Test: {test_acc_history[-1]:.4f}")

                        # Biểu đồ nhỏ gọn
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot(range(1, len(test_acc_history) + 1), test_acc_history, label="Test Acc")
                        ax.plot(range(1, len(valid_acc_history) + 1), valid_acc_history, label="Valid Acc")
                        ax.set_xlabel("Iteration")
                        ax.set_ylabel("Accuracy")
                        ax.legend()
                        st.pyplot(fig)

                        # Hiển thị 5 mẫu thay vì 10 để tiết kiệm tài nguyên
                        st.subheader("5 mẫu ví dụ từ tập Test")
                        random_indices = np.random.choice(len(X_test), 5, replace=False)
                        X_samples, y_true = X_test[random_indices], y_test[random_indices]
                        with torch.no_grad():
                            y_pred = torch.argmax(model(torch.tensor(X_samples, dtype=torch.float32)), dim=1).numpy()

                        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
                        for i, ax in enumerate(axes):
                            ax.imshow(X_samples[i].reshape(28, 28), cmap='gray')
                            ax.set_title(f"Thực: {y_true[i]}\nDự đoán: {y_pred[i]}")
                            ax.axis('off')
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Lỗi trong quá trình huấn luyện: {e}")

# Tab 1: Lý thuyết (giữ nguyên nội dung, tối ưu định dạng)
with tab1:
    st.title("Hiểu Biết về Pseudo-Labeling")
    st.header("1. Pseudo-Labeling là gì?")
    st.write("Pseudo-Labeling là kỹ thuật học bán giám sát tận dụng dữ liệu không nhãn bằng cách dự đoán nhãn giả và thêm vào tập có nhãn.")

    st.header("2. Tại sao cần Pseudo-Labeling?")
    st.write("- Ít dữ liệu có nhãn.\n- Cải thiện hiệu suất.\n- Ứng dụng thực tế như phân loại MNIST.")

    st.header("3. Quy trình Pseudo-Labeling")
    st.latex(r"L = \{(x_i, y_i)\}, U = \{x_j\}")
    st.write("1. Huấn luyện trên \(L\).\n2. Dự đoán nhãn giả trên \(U\).\n3. Lọc bằng ngưỡng.\n4. Cập nhật \(L\) và \(U\).\n5. Lặp lại.")

    st.header("4. Ưu điểm và Hạn chế")
    st.write("**Ưu điểm**: Đơn giản, hiệu quả.\n**Hạn chế**: Nhạy với nhiễu, phụ thuộc ngưỡng.")

# Tab 3: Dự đoán
with tab3:
    def preprocess_image(image, source="upload"):
        if source == "upload":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = np.array(image)[:, :, 0]  # Từ canvas
        image = cv2.resize(image, (28, 28)).reshape(1, -1) / 255.0
        return image

    if "model" not in st.session_state:
        st.error("Vui lòng huấn luyện mô hình trước!")
    else:
        st.header("Dự đoán chữ số viết tay")
        option = st.radio("Chọn phương thức nhập:", ["Tải ảnh lên", "Vẽ số"])

        if option == "Tải ảnh lên":
            uploaded_file = st.file_uploader("Tải ảnh (PNG, JPG)", type=["png", "jpg", "jpeg"])
            if uploaded_file:
                image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
                processed_image = preprocess_image(image, "upload")
                st.image(image, caption="Ảnh tải lên", width=200)

                if st.button("Dự đoán"):
                    with torch.no_grad():
                        model = st.session_state.model
                        model.eval()
                        outputs = model(torch.tensor(processed_image, dtype=torch.float32))
                        probs = torch.softmax(outputs, dim=1).numpy()[0]
                        pred = np.argmax(probs)
                        st.write(f"Dự đoán: {pred}, Độ tin cậy: {probs[pred] * 100:.2f}%")

        else:
            canvas_result = st_canvas(fill_color="rgba(255, 255, 255, 0.0)", stroke_width=15, 
                                     stroke_color="white", background_color="black", 
                                     width=280, height=280, drawing_mode="freedraw", key="canvas")
            if st.button("Dự đoán") and canvas_result.image_data is not None:
                processed_image = preprocess_image(canvas_result.image_data, "canvas")
                with torch.no_grad():
                    model = st.session_state.model
                    model.eval()
                    outputs = model(torch.tensor(processed_image, dtype=torch.float32))
                    probs = torch.softmax(outputs, dim=1).numpy()[0]
                    pred = np.argmax(probs)
                    st.write(f"Dự đoán: {pred}, Độ tin cậy: {probs[pred] * 100:.2f}%")

# Tab 4: MLflow
with tab4:
    st.header("Tracking MLflow")
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        experiments = mlflow.search_experiments()

        if experiments:
            st.write("Danh sách thí nghiệm")
            df_experiments = pd.DataFrame([{"ID": exp.experiment_id, "Name": exp.name} for exp in experiments])
            st.dataframe(df_experiments)

            selected_exp_name = st.selectbox("Chọn thí nghiệm", [exp.name for exp in experiments])
            selected_exp_id = next(exp.experiment_id for exp in experiments if exp.name == selected_exp_name)
            runs = mlflow.search_runs(selected_exp_id)

            if not runs.empty:
                st.write("Danh sách runs")
                st.dataframe(runs[["run_id", "start_time"]])
                selected_run_id = st.selectbox("Chọn run", runs["run_id"])
                run = mlflow.get_run(selected_run_id)
                st.write(f"Run ID: {run.info.run_id}")
                st.write("Metrics:", run.data.metrics)
                st.write("Params:", run.data.params)
            else:
                st.warning("Không có runs trong thí nghiệm này.")
        else:
            st.warning("Không có thí nghiệm nào.")
    except Exception as e:
        st.error(f"Lỗi khi truy cập MLflow: {e}")
