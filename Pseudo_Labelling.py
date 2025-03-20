import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
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

# Cache dữ liệu MNIST
@st.cache_data
def load_mnist(sample_size):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    if sample_size < mnist.data.shape[0]:
        X, _, y, _ = train_test_split(X, y, train_size=sample_size / mnist.data.shape[0], random_state=42, stratify=y)
    return X, y

# Cache mô hình Neural Network
@st.cache_resource
def create_model(num_hidden_layers, hidden_size, activation):
    class SimpleNN(nn.Module):
        def __init__(self):
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
    return SimpleNN()

# Tiêu đề ứng dụng
st.title("Phân loại chữ số viết tay MNIST với Self-Training Neural Network")

# Tạo các tab
tab1, tab2, tab3, tab4 = st.tabs(["Lý thuyết", "Huấn luyện", "Dự Đoán", "MLflow"])

# Tab 1: Lý thuyết
with tab1:
    st.title(":brain: Hiểu Biết về Pseudo-Labeling trong Học Bán Giám Sát")

    st.header(":book: 1. Pseudo-Labeling là gì?")
    st.write("""
    :information_source: Pseudo-Labeling là một kỹ thuật học bán giám sát nhằm tận dụng dữ liệu không nhãn (unlabeled data) bằng cách:
    - Sử dụng mô hình được huấn luyện trên dữ liệu có nhãn để dự đoán nhãn cho dữ liệu không nhãn.
    - Chọn các nhãn dự đoán (pseudo-labels) đủ tự tin (dựa trên ngưỡng) để thêm vào tập dữ liệu có nhãn.
    - Huấn luyện lại mô hình trên tập dữ liệu mở rộng.
    """)

    st.header(":question: 2. Tại sao cần Pseudo-Labeling?")
    st.write("""
    :star: **Dữ liệu có nhãn ít**: Thu thập nhãn tốn kém, trong khi dữ liệu không nhãn thường dồi dào.  
    :star: **Cải thiện hiệu suất**: Tận dụng dữ liệu không nhãn để tăng độ chính xác của mô hình.  
    :star: **Ứng dụng thực tế**: Ví dụ: phân loại ảnh (như MNIST) khi chỉ có một phần nhỏ dữ liệu được gắn nhãn.
    """)

    st.header(":gear: 3. Quy trình Pseudo-Labeling trong Self-Training")
    st.write(":memo: Dưới đây là các bước cơ bản của Pseudo-Labeling với công thức minh họa:")

    st.subheader("Bước 1: Chuẩn bị dữ liệu")
    st.write("Tập Labeled (L): Dữ liệu có nhãn ban đầu:")
    st.latex(r"L = \{(x_i, y_i)\}_{i=1}^{N_L}")
    st.write("Tập Unlabeled (U): Dữ liệu không nhãn:")
    st.latex(r"U = \{x_j\}_{j=1}^{N_U}")
    
    st.subheader("Bước 2: Huấn luyện mô hình ban đầu")
    st.write("Dùng tập \( L \) để huấn luyện mô hình \( f(x; \theta) \):")
    st.latex(r"\min_{\theta} \sum_{(x_i, y_i) \in L} \text{Loss}(f(x_i; \theta), y_i)")
    
    st.subheader("Bước 3: Dự đoán nhãn giả")
    st.write("Dự đoán trên tập \( U \) bằng \( f(x; \theta) \):")
    st.latex(r"y_{pseudo,j} = \arg\max_{k} (p_j(k))")
    
    st.subheader("Bước 4: Lọc bằng ngưỡng")
    st.write("Chọn mẫu nếu xác suất tối đa vượt ngưỡng \( \tau \):")
    st.latex(r"\max_{k} (p_j(k)) \geq \tau")
    
    st.subheader("Bước 5: Cập nhật tập dữ liệu")
    st.write("Thêm mẫu được chọn vào \( L \):")
    st.latex(r"L = L \cup \{(x_j, y_{pseudo,j})\}")
    st.write("Loại mẫu khỏi \( U \):")
    st.latex(r"U = U \setminus \{x_j\}")
    
    st.subheader("Bước 6: Lặp lại")
    st.write("- Huấn luyện lại \( f(x; \theta) \) trên \( L \) mới.")
    st.write("- Lặp lại từ Bước 3 cho đến khi \( U = \emptyset \) hoặc đạt số vòng lặp tối đa.")
    
    st.header("4. Ưu điểm và Hạn chế")
    st.subheader(":thumbsup: Ưu điểm:")
    st.write("""
    - :zap: Đơn giản, dễ triển khai.  
    - :rocket: Tận dụng dữ liệu không nhãn hiệu quả.  
    - :chart_with_upwards_trend: Cải thiện độ chính xác khi dữ liệu có nhãn ít.
    """)
    
    st.subheader(":thumbsdown: Hạn chế:")
    st.write("""
    - :warning: Nhạy cảm với nhiễu: Nhãn giả sai có thể làm giảm chất lượng mô hình.  
    - :scales: Phụ thuộc ngưỡng: Ngưỡng cao → ít nhãn giả, ngưỡng thấp → nhiều nhãn sai.  
    - :muscle: Yêu cầu mô hình ban đầu tốt để dự đoán chính xác.
    """)
    
    st.header(":tada: 5. Kết luận")
    st.write("""
    :light_bulb: Pseudo-Labeling là một kỹ thuật mạnh mẽ trong học bán giám sát, đặc biệt khi bạn có ít dữ liệu có nhãn. Hiệu quả của nó phụ thuộc vào ngưỡng \( \tau \), chất lượng mô hình ban đầu \( f(x; \theta) \), và cách cấu hình quá trình lặp.
    """)

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
    # Tiêu đề cho phần tham số Neural Network
    st.subheader("Tham số mạng Neural Network")
    num_epochs = st.number_input("Số epochs mỗi vòng", min_value=1, max_value=50, value=10)
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128, 256], index=1)
    learning_rate = st.number_input("Tốc độ học", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001)
    num_hidden_layers = st.number_input("Số lớp ẩn", min_value=1, max_value=100, value=1)
    hidden_neurons = st.selectbox("Số nơ-ron mỗi lớp ẩn", [16, 32, 64, 128, 256], index=1)
    activation_function = st.selectbox("Hàm kích hoạt", ["ReLU", "Sigmoid", "Tanh"], index=0)

    # Tiêu đề cho phần tham số Pseudo Labeling
    st.subheader("Tham số gán nhãn giả (Pseudo Labeling)")
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
                st.write("### Kết quả cuối cùng:")
                st.write(f"- **Độ chính xác trên Validation**: {valid_acc_history[-1]:.4f}")
                st.write(f"- **Độ chính xác trên Test**: {test_acc_history[-1]:.4f}")

                # Hiển thị biểu đồ tiến trình
                st.subheader("Tiến trình Self-Training")
                fig, ax = plt.subplots()
                ax.plot(range(1, len(test_acc_history) + 1), test_acc_history, label="Test Accuracy")
                ax.plot(range(1, len(valid_acc_history) + 1), valid_acc_history, label="Validation Accuracy")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Accuracy")
                ax.legend()
                st.pyplot(fig)

                # Hiển thị 10 mẫu ví dụ từ tập Test sau khi huấn luyện
                st.subheader("10 mẫu ví dụ từ tập Test với dự đoán")
                num_examples = 10  # Số lượng mẫu muốn hiển thị
                random_indices = np.random.choice(len(X_test), num_examples, replace=False)
                X_samples = X_test[random_indices]
                y_true = y_test[random_indices]

                # Dự đoán nhãn bằng mô hình đã huấn luyện
                model.eval()
                X_samples_tensor = torch.tensor(X_samples, dtype=torch.float32)
                with torch.no_grad():
                    outputs = model(X_samples_tensor)
                    y_pred = torch.argmax(outputs, dim=1).numpy()

                # Tạo figure để hiển thị các mẫu (2 hàng, mỗi hàng 5 mẫu)
                fig, axes = plt.subplots(2, 5, figsize=(10, 4))
                for i, (sample, true_label, pred_label) in enumerate(zip(X_samples, y_true, y_pred)):
                    row = i // 5
                    col = i % 5
                    image = sample.reshape(28, 28)
                    axes[row, col].imshow(image, cmap='gray')
                    axes[row, col].set_title(f"Thực: {true_label}\nDự đoán: {pred_label}")
                    axes[row, col].axis('off')
                plt.tight_layout()
                st.pyplot(fig)

    if st.button("Bắt đầu Self-Training"):
        run_self_training()

# Tab 3: Dự đoán
with tab3:
    def preprocess_uploaded_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28))
        image = image.reshape(1, -1) / 255.0
        return image

    def preprocess_canvas_image(image_data):
        image = np.array(image_data)[:, :, 0]
        image = cv2.resize(image, (28, 28))
        image = image.reshape(1, -1) / 255.0
        return image

    if "model" not in st.session_state:
        st.error("⚠️ Mô hình chưa được huấn luyện! Hãy quay lại tab 'Huấn luyện' để huấn luyện trước.")
        st.stop()

    st.header("🖍️ Dự đoán chữ số viết tay")
    option = st.radio("🖼️ Chọn phương thức nhập:", ["📂 Tải ảnh lên", "✏️ Vẽ số"])

    # Fragment cho dự đoán
    @st.fragment
    def predict_image():
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
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=15,
                stroke_color="white",
                background_color="black",
                width=280,
                height=280,
                drawing_mode="freedraw",
                key="canvas"
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

    predict_image()

# Tab 4: MLflow
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
