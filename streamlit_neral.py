import datetime
import random
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

# Tiêu đề ứng dụng
st.title("Phân loại chữ số viết tay MNIST với Neural_Netwwork")

# Tạo các tab
tab1, tab2, tab3 = st.tabs([
    "Lý thuyết",
    "Huấn luyện",
    "MLflow"
])

# Tab 1: Lý thuyết
with tab1:
    st.header("Hướng dẫn: Lý thuyết tổng quát về mạng nơ-ron")
    st.markdown("""
    Mạng nơ-ron nhân tạo (Artificial Neural Networks - ANN) là một mô hình học máy được lấy cảm hứng từ cách hoạt động của não bộ con người. Nó được thiết kế để học hỏi và dự đoán từ dữ liệu thông qua các lớp nơ-ron kết nối với nhau. Dưới đây là các khái niệm và bước hoạt động tổng quát:
    """)

    # Phần 1: Cấu trúc cơ bản
    st.markdown("""
    ### 1. Cấu trúc cơ bản
    - **Nơ-ron (Neuron)**: Đơn vị tính toán cơ bản, nhận đầu vào, xử lý, và tạo đầu ra.
    - **Lớp (Layers)**:
      - **Lớp đầu vào (Input Layer)**: Nhận dữ liệu thô (ví dụ: hình ảnh, số liệu).
      - **Lớp ẩn (Hidden Layers)**: Xử lý dữ liệu để học các đặc trưng phức tạp.
      - **Lớp đầu ra (Output Layer)**: Đưa ra kết quả cuối cùng (ví dụ: phân loại, dự đoán số).
    - **Trọng số (Weights)** và **Bias**: Các tham số điều chỉnh mức độ ảnh hưởng của đầu vào, được cập nhật trong quá trình học.
    """)
    st.image("https://miro.medium.com/max/1200/1*FYiM8SggQTVQz_Hrmz6fOw.png", 
             caption="Cấu trúc cơ bản của mạng nơ-ron: Lớp đầu vào, lớp ẩn, và lớp đầu ra.", width=300)

    # Phần 2: Cách hoạt động (chi tiết từng bước với ảnh mới)
    st.markdown("""
    ### 2. Cách hoạt động
    Mạng nơ-ron hoạt động thông qua một chuỗi các bước tuần tự, từ việc nhận dữ liệu, xử lý, dự đoán, đến điều chỉnh để cải thiện. Dưới đây là các bước nguyên lý hoạt động chi tiết:
    """)

    st.markdown("""
    #### Bước 1: Nhận và truyền dữ liệu đầu vào
    - Dữ liệu thô (ví dụ: hình ảnh, số liệu) được đưa vào lớp đầu vào.
    - Mỗi nơ-ron trong lớp đầu vào đại diện cho một giá trị của dữ liệu (ví dụ: một pixel trong ảnh).
    - Dữ liệu sau đó được truyền đến lớp ẩn đầu tiên thông qua các kết nối có trọng số.
    """)
    st.image("https://i.imgur.com/8g6zK9U.png", 
             caption="Dữ liệu đầu vào được đưa vào lớp đầu tiên.", width=350)

    st.markdown("""
    #### Bước 2: Tính tổng trọng số tại nơ-ron
    - Tại mỗi nơ-ron trong lớp ẩn, dữ liệu đầu vào được nhân với trọng số tương ứng và cộng với bias:
      $$ z = W \\cdot X + b $$
      - \(W\): Ma trận trọng số (weights).
      - \(X\): Vector dữ liệu đầu vào (inputs).
      - \(b\): Giá trị bias (điều chỉnh).
    - \(z\) là tổng trọng số, đại diện cho giá trị chưa qua xử lý của nơ-ron.
    """)
    st.image("https://i.imgur.com/5p5gXZm.png", 
             caption="Tính tổng trọng số tại một nơ-ron.", width=350)

    st.markdown("""
    #### Bước 3: Áp dụng hàm kích hoạt
    - Tổng trọng số \(z\) được truyền qua một hàm kích hoạt (activation function) để tạo tính phi tuyến:
      - **ReLU**: \( a = \\max(0, z) \) (chỉ giữ giá trị dương).
      - **Sigmoid**: \( a = \\frac{1}{1 + e^{-z}} \) (giới hạn đầu ra từ 0 đến 1).
      - **Tanh**: \( a = \\tanh(z) \) (giới hạn đầu ra từ -1 đến 1).
    - Đầu ra \(a\) của hàm kích hoạt là giá trị cuối cùng của nơ-ron, được truyền sang lớp tiếp theo.
    """)
    st.image("https://miro.medium.com/max/1200/1*XxxiA0jJvPrHEJHD4z893g.png", 
             caption="Áp dụng hàm kích hoạt (ReLU, Sigmoid, Tanh).", width=400)

    st.markdown("""
    #### Bước 4: Lan truyền qua các lớp
    - Đầu ra của lớp trước (sau khi qua hàm kích hoạt) trở thành đầu vào của lớp tiếp theo.
    - Quá trình tính tổng trọng số và áp dụng hàm kích hoạt lặp lại qua tất cả các lớp ẩn, đến lớp đầu ra.
    - Lớp đầu ra tạo ra dự đoán cuối cùng của mô hình (ví dụ: xác suất phân loại).
    """)
    st.image("https://i.imgur.com/Z4N3g5M.png", 
             caption="Lan truyền qua các lớp từ đầu vào đến đầu ra.", width=400)

    st.markdown("""
    #### Bước 5: Tính hàm mất mát
    - So sánh dự đoán của mô hình với giá trị thực tế để đo sai số (loss).
    - Ví dụ hàm mất mát:
      - **Mean Squared Error (MSE)**: \( L = \\frac{1}{n} \\sum (y - \\hat{y})^2 \) (cho hồi quy).
      - **Cross-Entropy Loss**: \( L = -\\frac{1}{n} \\sum [y \\cdot \\log(\\hat{y})] \) (cho phân loại).
    - \(y\): Giá trị thực tế, \(\\hat{y}\): Dự đoán.
    """)
    st.image("https://i.imgur.com/3XzJ4gq.png", 
             caption="Tính hàm mất mát để đo sai số.", width=350)

    st.markdown("""
    #### Bước 6: Tính gradient bằng lan truyền ngược
    - Dùng quy tắc chuỗi (chain rule) để tính gradient của hàm mất mát theo từng trọng số và bias:
      $$ \\frac{\\partial L}{\\partial W}, \\frac{\\partial L}{\\partial b} $$
    - Gradient chỉ ra hướng và mức độ thay đổi cần thiết để giảm sai số.
    """)
    st.image("https://i.imgur.com/8Q4f5vR.png", 
             caption="Lan truyền ngược: Tính gradient để điều chỉnh.", width=400)

    st.markdown("""
    #### Bước 7: Cập nhật trọng số
    - Sử dụng thuật toán tối ưu (ví dụ: Gradient Descent) để điều chỉnh trọng số và bias:
      $$ W = W - \\eta \\cdot \\frac{\\partial L}{\\partial W} $$
      $$ b = b - \\eta \\cdot \\frac{\\partial L}{\\partial b} $$
    - \(\\eta\): Tốc độ học (learning rate), quyết định bước cập nhật lớn hay nhỏ.
    """)
    st.image("https://i.imgur.com/5n5sX7v.png", 
             caption="Cập nhật trọng số bằng Gradient Descent.", width=350)

    st.markdown("""
    #### Bước 8: Lặp lại quá trình huấn luyện
    - Lặp qua toàn bộ dữ liệu nhiều lần (epochs), chia thành các batch nhỏ để cập nhật trọng số dần dần.
    - Sau mỗi lần lặp, mô hình cải thiện khả năng dự đoán bằng cách giảm hàm mất mát.
    """)
    st.image("https://i.imgur.com/6g6K9vN.png", 
             caption="Lặp lại quá trình huấn luyện qua nhiều epochs.", width=400)

    # Phần 3: Vai trò của các thành phần
    st.markdown("""
    ### 3. Vai trò của các thành phần
    - **Hàm kích hoạt**: Tạo tính phi tuyến, giúp mô hình học các đặc trưng phức tạp.
    - **Tốc độ học (Learning Rate)**: Quyết định bước cập nhật trọng số, ảnh hưởng đến tốc độ và độ ổn định.
    - **Số lớp và nơ-ron**: Tăng độ phức tạp của mô hình, nhưng cần cân bằng để tránh overfitting hoặc underfitting.
    """)

    # Phần 4: Ứng dụng
    st.markdown("""
    ### 4. Ứng dụng
    - **Phân loại**: Nhận diện hình ảnh, văn bản (ví dụ: chữ số viết tay).
    - **Hồi quy**: Dự đoán giá trị liên tục (ví dụ: giá nhà).
    - **Xử lý ngôn ngữ tự nhiên, thị giác máy tính**: Dùng mạng sâu (Deep Neural Networks).
    """)

    # Phần 5: Khái niệm quan trọng
    st.markdown("""
    ### 5. Một số khái niệm quan trọng
    - **Overfitting**: Mô hình học quá tốt trên dữ liệu huấn luyện, nhưng kém trên dữ liệu mới.
    - **Underfitting**: Mô hình không học đủ, dự đoán kém trên cả dữ liệu huấn luyện.
    - **Regularization**: Kỹ thuật (như Dropout) để giảm overfitting.
    """)

    # Thêm script MathJax để hiển thị công thức toán học
    st.markdown("""
    <script type="text/javascript" async
      src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
    </script>
    """, unsafe_allow_html=True)

# Tab Huấn luyện
with tab2:  # Giữ nguyên 'with tab2:' nếu bạn đang dùng tab
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
        help="Số lượng mẫu dữ liệu được lấy từ MNIST (tối đa 70,000). Giá trị lớn hơn sẽ tăng độ chính xác nhưng cần nhiều thời gian huấn luyện hơn."
    )
    
    # Chọn tỷ lệ tập Test và Validation
    test_size = st.slider(
        "Chọn tỷ lệ dữ liệu Test", 
        0.1, 0.5, 0.2, 0.05,
        help="Tỷ lệ dữ liệu dùng để kiểm tra mô hình (10%-50%). Nên chọn khoảng 20%-30% để đánh giá hiệu quả mà không làm giảm dữ liệu huấn luyện."
    )
    valid_size = st.slider(
        "Chọn tỷ lệ dữ liệu Validation từ Train", 
        0.1, 0.3, 0.2, 0.05,
        help="Tỷ lệ dữ liệu từ tập Train dùng để kiểm tra trong lúc huấn luyện (10%-30%). Giúp điều chỉnh mô hình mà không dùng tập Test."
    )

    if st.button("Chia tách dữ liệu"):
        mnist = st.session_state.mnist_data
        X, y = mnist.data / 255.0, mnist.target.astype(int)
        
        if sample_size < st.session_state.total_samples:
            X, _, y, _ = train_test_split(X, y, train_size=sample_size, random_state=42, stratify=y)
        
        # Lưu dữ liệu gốc vào session_state
        st.session_state.X = X
        st.session_state.y = y

        # Chia dữ liệu thành Train_full và Test
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            st.session_state.X, st.session_state.y, test_size=test_size, random_state=42, stratify=st.session_state.y
        )
        # Chia Train_full thành Train và Validation
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
        st.session_state.mnist_loaded = True

        # Hiển thị thông tin sau khi chia tách
        st.write(f"Dữ liệu đã được chia tách với {sample_size} mẫu!")
        st.write(f"- Dữ liệu Train: {st.session_state.X_train.shape} ({(1-test_size)*(1-valid_size)*100:.1f}%)")
        st.write(f"- Dữ liệu Validation: {st.session_state.X_valid.shape} ({(1-test_size)*valid_size*100:.1f}%)")
        st.write(f"- Dữ liệu Test: {st.session_state.X_test.shape} ({test_size*100:.1f}%)")

    # Hiển thị ví dụ hình ảnh
    st.subheader("Ví dụ hình ảnh từ tập Train")
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

    # Tab Huấn luyện
    st.header("Huấn luyện Neural Network")

    # Các tham số cơ bản thông dụng cho người dùng lựa chọn
    st.subheader("Cấu hình huấn luyện")
    num_epochs = st.number_input(
        "Số epochs", 
        min_value=1, 
        max_value=50, 
        value=10,
        help="Số lần mô hình học qua toàn bộ dữ liệu. Tăng giá trị để học tốt hơn, nhưng quá nhiều có thể gây overfitting."
    )
    batch_size = st.selectbox(
        "Batch size", 
        [16, 32, 64, 128, 256, 512], 
        index=1,  # Mặc định 32
        help="Số mẫu xử lý cùng lúc. Giá trị nhỏ tăng độ chính xác nhưng chậm hơn; giá trị lớn tăng tốc độ nhưng cần bộ nhớ lớn."
    )
    learning_rate = st.number_input(
        "Tốc độ học (learning rate)", 
        min_value=0.0001, 
        max_value=0.1, 
        value=0.001, 
        step=0.0001,
        help="Kiểm soát tốc độ học của mô hình. Giá trị nhỏ học chậm nhưng ổn định; giá trị lớn học nhanh nhưng có thể không hội tụ."
    )
    hidden_neurons = st.selectbox(
        "Số nơ-ron lớp ẩn", 
        [32, 64, 128, 256, 512], 
        index=2,  # Mặc định 128
        help="Số nơ-ron trong lớp ẩn. Giá trị lớn tăng khả năng học đặc trưng phức tạp, nhưng quá nhiều có thể gây overfitting."
    )
    activation_function = st.selectbox(
        "Hàm kích hoạt (Activation Function)",
        ["ReLU", "Sigmoid", "Tanh"],
        index=0,  # Mặc định ReLU
        help="Hàm biến đổi đầu ra của lớp ẩn. ReLU phổ biến và nhanh; Sigmoid phù hợp với giá trị 0-1; Tanh cân bằng quanh 0."
    )

    # Nhập tên cho thí nghiệm MLflow
    experiment_name = st.text_input(
        "Nhập tên cho thí nghiệm MLflow", 
        value="",
        help="Tên để lưu thí nghiệm trong MLflow. Nếu để trống, hệ thống sẽ tự tạo tên dựa trên thời gian."
    )
    if not experiment_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"Neural_Network_MNIST_{timestamp}"
    
    # Nút để bắt đầu huấn luyện
    if st.button("Huấn luyện mô hình"):
        # Kiểm tra dữ liệu đã được chia tách chưa
        if not st.session_state.get("data_split_done", False):
            st.error("Vui lòng chia tách dữ liệu trước!")
        else:
            # Lấy dữ liệu từ session state
            X_train = st.session_state.X_train
            y_train = st.session_state.y_train
            X_valid = st.session_state.X_valid
            y_valid = st.session_state.y_valid
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

            # Chuyển dữ liệu sang tensor
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
            y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)

            # Tạo DataLoader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Định nghĩa mô hình Neural Network với activation tùy chỉnh
            class SimpleNN(nn.Module):
                def __init__(self, hidden_size=hidden_neurons, activation=activation_function):
                    super(SimpleNN, self).__init__()
                    self.fc1 = nn.Linear(784, hidden_size)  # Input: 784, Hidden: tùy chỉnh
                    # Chọn hàm kích hoạt dựa trên lựa chọn của người dùng
                    if activation == "ReLU":
                        self.activation = nn.ReLU()
                    elif activation == "Sigmoid":
                        self.activation = nn.Sigmoid()
                    elif activation == "Tanh":
                        self.activation = nn.Tanh()
                    self.fc2 = nn.Linear(hidden_size, 10)   # Hidden: tùy chỉnh, Output: 10

                def forward(self, x):
                    x = self.fc1(x)
                    x = self.activation(x)
                    x = self.fc2(x)
                    return x

            # Khởi tạo mô hình, loss và optimizer
            model = SimpleNN(hidden_size=hidden_neurons, activation=activation_function)
            criterion = nn.CrossEntropyLoss()           # Cố định CrossEntropyLoss
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Cố định Adam

            # Thiết lập MLflow
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run():
                # Log các tham số
                mlflow.log_param("num_epochs", num_epochs)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("learning_rate", learning_rate)
                mlflow.log_param("hidden_neurons", hidden_neurons)
                mlflow.log_param("activation_function", activation_function)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("valid_size", valid_size)

                # Thanh tiến trình và trạng thái
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Danh sách để lưu độ chính xác
                train_acc_history = []
                valid_acc_history = []

                # Huấn luyện mô hình
                for epoch in range(num_epochs):
                    model.train()
                    correct = 0
                    total = 0
                    for inputs, labels in train_loader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    train_acc = correct / total
                    train_acc_history.append(train_acc)

                    # Đánh giá trên tập validation
                    model.eval()
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

                    # Log metrics vào MLflow
                    mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                    mlflow.log_metric("valid_accuracy", valid_acc, step=epoch)

                    # Cập nhật thanh tiến trình và thông tin
                    progress = (epoch + 1) / num_epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch+1}/{num_epochs}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {valid_acc:.4f}")

                # Lưu mô hình vào MLflow
                mlflow.pytorch.log_model(model, "model")

                # Hiển thị thông báo hoàn tất
                st.success("Huấn luyện hoàn tất!")

                # Vẽ sơ đồ cấu trúc các lớp của mô hình với kích thước tỷ lệ
                st.subheader("Sơ đồ cấu trúc các lớp của mô hình")
                fig, ax = plt.subplots(figsize=(12, 5))

                # Định nghĩa kích thước các lớp
                model_dims = {
                    "Input Layer": 784,
                    f"Hidden Layer\n({activation_function})": hidden_neurons,
                    "Output Layer": 10
                }

                # Vị trí của các lớp trên trục x
                x_positions = [0, 3, 5]

                # Tính chiều cao tỷ lệ dựa trên số nơ-ron (log scale để tránh quá chênh lệch)
                max_height = 2.0  # Chiều cao tối đa của hình chữ nhật
                heights = [min(max_height, max_height * size / 784) for size in model_dims.values()]

                # Vẽ các lớp dưới dạng hình chữ nhật với chiều cao tỷ lệ
                for i, (layer_name, size) in enumerate(model_dims.items()):
                    rect = patches.Rectangle(
                        (x_positions[i] - 0.4, -heights[i]/2),  # Vị trí (x, y)
                        0.8, heights[i],  # Chiều rộng và chiều cao
                        linewidth=1, edgecolor='black', facecolor='lightblue'
                    )
                    ax.add_patch(rect)
                    ax.text(x_positions[i], heights[i]/2 + 0.2, f"{layer_name}\n{size} nơ-ron", 
                            ha='center', va='bottom', fontsize=12)

                # Vẽ mũi tên kết nối các lớp
                for i in range(len(x_positions) - 1):
                    ax.arrow(x_positions[i] + 0.4, 0, x_positions[i+1] - x_positions[i] - 0.8, 0, 
                             head_width=0.1, head_length=0.1, fc='black', ec='black')

                # Tùy chỉnh biểu đồ
                ax.set_xlim(-1, 6)
                ax.set_ylim(-max_height/2 - 0.5, max_height/2 + 0.8)
                ax.axis('off')  # Tắt trục để sơ đồ trông gọn gàng
                st.pyplot(fig)

                # Vẽ biểu đồ huấn luyện
                st.subheader("Biểu đồ độ chính xác qua các epoch")
                fig, ax = plt.subplots()
                ax.plot(range(1, num_epochs+1), train_acc_history, label='Train Accuracy')
                ax.plot(range(1, num_epochs+1), valid_acc_history, label='Validation Accuracy')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.legend()
                st.pyplot(fig)
# Tab 3: MLflow
with tab3:
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