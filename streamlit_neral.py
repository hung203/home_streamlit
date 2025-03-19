import datetime
import random
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
st.title("Phân loại chữ số viết tay MNIST với Neural Network")

# Tạo các tab
tab1, tab2, tab3, tab4 = st.tabs(["Lý thuyết", "Huấn luyện", "Dự Đoán", "MLflow"])

# Tab 1: Lý thuyết
with tab1:
    st.header("Hướng dẫn: Lý thuyết tổng quát về mạng nơ-ron 🧠")
    st.markdown("""
    Mạng nơ-ron nhân tạo (Artificial Neural Networks - ANN) là một mô hình học máy được lấy cảm hứng từ cách hoạt động của não bộ con người. Nó được thiết kế để học hỏi và dự đoán từ dữ liệu thông qua các lớp nơ-ron kết nối với nhau. Dưới đây là các khái niệm và bước hoạt động tổng quát:
    """)

    st.markdown("""
    ### 1. Cấu trúc cơ bản 🛠️
    - **Nơ-ron (Neuron)** ⚙️: Đơn vị tính toán cơ bản, nhận đầu vào, xử lý, và tạo đầu ra.
    - **Lớp (Layers)** 📚:
      - **Lớp đầu vào (Input Layer)** 📥: Nhận dữ liệu thô (ví dụ: hình ảnh, số liệu).
      - **Lớp ẩn (Hidden Layers)** 🕵️: Xử lý dữ liệu để học các đặc trưng phức tạp.
      - **Lớp đầu ra (Output Layer)** 📤: Đưa ra kết quả cuối cùng (ví dụ: phân loại, dự đoán số).
    - **Trọng số (Weights)** ⚖️ và **Bias** 🔧: Các tham số điều chỉnh mức độ ảnh hưởng của đầu vào, được cập nhật trong quá trình học.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Artificial_neural_network.svg/525px-Artificial_neural_network.svg.png", 
             caption="Cấu trúc cơ bản của mạng nơ-ron: Lớp đầu vào, lớp ẩn, và lớp đầu ra.", width=300)

    st.markdown("""
    ### 2. Cách hoạt động ⚡
    Mạng nơ-ron hoạt động thông qua một chuỗi các bước tuần tự, từ việc nhận dữ liệu, xử lý, dự đoán, đến điều chỉnh để cải thiện.
    """)

    st.markdown("""
    #### Bước 1: Nhận và truyền dữ liệu đầu vào 📡
    - Dữ liệu thô (ví dụ: hình ảnh, số liệu) được đưa vào lớp đầu vào.
    - Mỗi nơ-ron trong lớp đầu vào đại diện cho một giá trị của dữ liệu (ví dụ: một pixel trong ảnh).
    - Dữ liệu sau đó được truyền đến lớp ẩn đầu tiên thông qua các kết nối có trọng số.
    """)

    st.markdown("""
    #### Bước 2: Tính tổng trọng số tại nơ-ron ➕
    - Tại mỗi nơ-ron trong lớp ẩn, dữ liệu đầu vào được nhân với trọng số tương ứng và cộng với bias:
    """)
    st.markdown(r"$$ z = W \cdot X + b $$")
    st.markdown("""
    Trong đó:
    - $ W $: Ma trận trọng số (weights).
    - $ X $: Vector dữ liệu đầu vào (inputs).
    - $ b $: Giá trị bias (điều chỉnh).
    - $ z $: Tổng trọng số, đại diện cho giá trị chưa qua xử lý của nơ-ron.
    """)

    st.markdown("""
    #### Bước 3: Áp dụng hàm kích hoạt 🚀
    """)
    st.markdown(r"- **ReLU**: $$ a = \max(0, z) $$ (chỉ giữ giá trị dương) 📈")
    st.markdown(r"- **Sigmoid**: $$ a = \frac{1}{1 + e^{-z}} $$ (giới hạn đầu ra từ 0 đến 1) 🔢")
    st.markdown(r"- **Tanh**: $$ a = \tanh(z) $$ (giới hạn đầu ra từ -1 đến 1) 📉")
    
    st.markdown("""
    - Đầu ra $ a $ của hàm kích hoạt là giá trị cuối cùng của nơ-ron, được truyền sang lớp tiếp theo.
    """)
    st.image("https://miro.medium.com/max/1200/1*XxxiA0jJvPrHEJHD4z893g.png", 
             caption="Áp dụng hàm kích hoạt (ReLU, Sigmoid, Tanh).", width=400)

    st.markdown("""
    #### Bước 5: Tính hàm mất mát 📊
    - So sánh dự đoán của mô hình với giá trị thực tế để đo sai số (loss).
    - Ví dụ hàm mất mát:
    """)
    st.markdown(r"- **Mean Squared Error (MSE)**: $$ L = \frac{1}{n} \sum (y - \hat{y})^2 $$ (cho hồi quy)")
    st.markdown(r"- **Cross-Entropy Loss**: $$ L = -\frac{1}{n} \sum [y \cdot \log(\hat{y})] $$ (cho phân loại)")
    
    st.markdown("""
    #### Bước 6: Tính gradient bằng lan truyền ngược 🔄
    """)
    st.markdown(r"$$ \frac{\partial L}{\partial W}, \frac{\partial L}{\partial b} $$")

    st.markdown("""
    #### Bước 7: Cập nhật trọng số 🔧
    """)
    st.markdown(r"$$ W = W - \eta \cdot \frac{\partial L}{\partial W} $$")
    st.markdown(r"$$ b = b - \eta \cdot \frac{\partial L}{\partial b} $$")
    
    st.markdown("""
    Trong đó:
    - $ \eta $: Tốc độ học (learning rate), quyết định bước cập nhật lớn hay nhỏ.
    """)

    st.markdown("""
    #### Bước 8: Lặp lại quá trình huấn luyện 🔁
    - Lặp qua toàn bộ dữ liệu nhiều lần (epochs), chia thành các batch nhỏ để cập nhật trọng số dần dần.
    - Sau mỗi lần lặp, mô hình cải thiện khả năng dự đoán bằng cách giảm hàm mất mát.
    """)

# Tab Huấn luyện
with tab2:
    st.header("1. Chọn kích thước và chia tập dữ liệu")
    
    # Khởi tạo trạng thái dữ liệu (di chuyển fetch_openml vào hàm cache)
    if "mnist_loaded" not in st.session_state:
        st.session_state.total_samples = 70000  # MNIST có 70,000 mẫu
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
        X, y = load_mnist(sample_size)
        st.session_state.X = X
        st.session_state.y = y

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, test_size=valid_size, random_state=42, stratify=y_train_full
        )

        st.session_state.X_train = X_train
        st.session_state.X_valid = X_valid
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_valid = y_valid
        st.session_state.y_test = y_test
        st.session_state.data_split_done = True
        st.session_state.mnist_loaded = True

        st.write(f"Dữ liệu đã được chia tách với {sample_size} mẫu!")
        st.write(f"- Dữ liệu Train: {st.session_state.X_train.shape} ({(1-test_size)*(1-valid_size)*100:.1f}%)")
        st.write(f"- Dữ liệu Validation: {st.session_state.X_valid.shape} ({(1-test_size)*valid_size*100:.1f}%)")
        st.write(f"- Dữ liệu Test: {st.session_state.X_test.shape} ({test_size*100:.1f}%)")

    # Fragment cho hiển thị ví dụ hình ảnh
    @st.fragment
    def show_sample_images():
        if st.session_state.get("data_split_done", False):
            X = st.session_state.X_train
            y = st.session_state.y_train
            indices = random.sample(range(len(X)), 5)
            fig, axs = plt.subplots(1, 5, figsize=(12, 3))
            for i, idx in enumerate(indices):
                img = X[idx].reshape(28, 28)
                axs[i].imshow(img, cmap='gray')
                axs[i].axis('off')
                axs[i].set_title(f"Label: {y[idx]}")
            st.pyplot(fig)

    st.subheader("Ví dụ hình ảnh từ tập Train")
    show_sample_images()

    st.header("Huấn luyện Neural Network")
    st.subheader("Cấu hình huấn luyện")
    num_epochs = st.number_input(
        "Số epochs", 
        min_value=1, 
        max_value=50, 
        value=10,
        help="Số lần mô hình học qua toàn bộ dữ liệu."
    )
    batch_size = st.selectbox(
        "Batch size", 
        [16, 32, 64, 128, 256, 512], 
        index=1,
        help="Số mẫu xử lý cùng lúc."
    )
    learning_rate = st.number_input(
        "Tốc độ học (learning rate)", 
        min_value=0.0001, 
        max_value=0.1, 
        value=0.001, 
        step=0.0001,
        help="Kiểm soát tốc độ học của mô hình."
    )
    num_hidden_layers = st.number_input(
        "Số lớp ẩn", 
        min_value=1, 
        max_value=20, 
        value=1, 
        step=1,
        help="Số lượng lớp ẩn trong mạng nơ-ron."
    )
    hidden_neurons = st.selectbox(
        "Số nơ-ron mỗi lớp ẩn", 
        [32, 64, 128, 256, 512], 
        index=2,
        help="Số nơ-ron trong mỗi lớp ẩn."
    )
    activation_function = st.selectbox(
        "Hàm kích hoạt (Activation Function)",
        ["ReLU", "Sigmoid", "Tanh"],
        index=0,
        help="Hàm biến đổi đầu ra của lớp ẩn."
    )

    experiment_name = st.text_input(
        "Nhập tên cho thí nghiệm MLflow", 
        value="",
        help="Tên để lưu thí nghiệm trong MLflow."
    )
    if not experiment_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"Neural_Network_MNIST_{timestamp}"
    
    # Fragment cho huấn luyện
    @st.fragment
    def train_and_evaluate():
        if not st.session_state.get("data_split_done", False):
            st.error("Vui lòng chia tách dữ liệu trước!")
        else:
            X_train = st.session_state.X_train
            y_train = st.session_state.y_train
            X_valid = st.session_state.X_valid
            y_valid = st.session_state.y_valid
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
            y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            model = create_model(num_hidden_layers, hidden_neurons, activation_function)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            mlflow.set_experiment(experiment_name)
            with mlflow.start_run() as run:
                mlflow.log_param("num_epochs", num_epochs)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("learning_rate", learning_rate)
                mlflow.log_param("num_hidden_layers", num_hidden_layers)
                mlflow.log_param("hidden_neurons", hidden_neurons)
                mlflow.log_param("activation_function", activation_function)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("valid_size", valid_size)
                mlflow.log_param("sample_size", sample_size)

                progress_bar = st.progress(0)
                status_text = st.empty()

                train_acc_history = []
                valid_acc_history = []
                test_acc_history = []

                for epoch in range(num_epochs):
                    model.train()
                    correct = 0
                    total = 0
                    train_loss = 0
                    for inputs, labels in train_loader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    train_acc = correct / total
                    train_loss = train_loss / len(train_loader)
                    train_acc_history.append(train_acc)

                    model.eval()
                    correct = 0
                    total = 0
                    valid_loss = 0
                    with torch.no_grad():
                        for inputs, labels in valid_loader:
                            outputs = model(inputs)
                            valid_loss += criterion(outputs, labels).item()
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    valid_acc = correct / total
                    valid_loss = valid_loss / len(valid_loader)
                    valid_acc_history.append(valid_acc)

                    correct = 0
                    total = 0
                    test_loss = 0
                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            outputs = model(inputs)
                            test_loss += criterion(outputs, labels).item()
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    test_acc = correct / total
                    test_loss = test_loss / len(test_loader)
                    test_acc_history.append(test_acc)

                    mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("valid_accuracy", valid_acc, step=epoch)
                    mlflow.log_metric("valid_loss", valid_loss, step=epoch)
                    mlflow.log_metric("test_accuracy", test_acc, step=epoch)
                    mlflow.log_metric("test_loss", test_loss, step=epoch)

                    progress = (epoch + 1) / num_epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}, Test Acc: {test_acc:.4f}")

                mlflow.pytorch.log_model(model, "model")
                st.session_state.model = model
                st.session_state.run_id = run.info.run_id

                st.success("Huấn luyện hoàn tất! Kết quả đã được log vào MLflow.")

                st.subheader("Sơ đồ cấu trúc các lớp của mô hình")
                fig, ax = plt.subplots(figsize=(12, 5))
                model_dims = {"Input Layer": 784}
                for i in range(num_hidden_layers):
                    model_dims[f"Hidden Layer {i+1}\n({activation_function})"] = hidden_neurons
                model_dims["Output Layer"] = 10
                x_positions = np.linspace(0, 5, num_hidden_layers + 2)
                max_height = 2.0
                heights = [min(max_height, max_height * size / 784) for size in model_dims.values()]
                for i, (layer_name, size) in enumerate(model_dims.items()):
                    rect = patches.Rectangle(
                        (x_positions[i] - 0.4, -heights[i]/2), 0.8, heights[i],
                        linewidth=1, edgecolor='black', facecolor='lightblue'
                    )
                    ax.add_patch(rect)
                    ax.text(x_positions[i], heights[i]/2 + 0.2, f"{layer_name}\n{size} nơ-ron", 
                            ha='center', va='bottom', fontsize=12)
                for i in range(len(x_positions) - 1):
                    ax.arrow(x_positions[i] + 0.4, 0, x_positions[i+1] - x_positions[i] - 0.8, 0, 
                             head_width=0.1, head_length=0.1, fc='black', ec='black')
                ax.set_xlim(-1, 6)
                ax.set_ylim(-max_height/2 - 0.5, max_height/2 + 0.8)
                ax.axis('off')
                st.pyplot(fig)

                st.subheader("Biểu đồ độ chính xác qua các epoch")
                fig, ax = plt.subplots()
                ax.plot(range(1, num_epochs+1), train_acc_history, label='Train Accuracy')
                ax.plot(range(1, num_epochs+1), valid_acc_history, label='Validation Accuracy')
                ax.plot(range(1, num_epochs+1), test_acc_history, label='Test Accuracy')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.legend()
                st.pyplot(fig)

    if st.button("Huấn luyện mô hình"):
        train_and_evaluate()

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
