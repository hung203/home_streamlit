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

# Tiêu đề ứng dụng
st.title("Phân loại chữ số viết tay MNIST với Streamlit và MLflow")

# Bước 1: Xử lý dữ liệu
st.header("1. Xử lý dữ liệu")
st.write("Đang tải dữ liệu MNIST...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data / 255.0, mnist.target.astype(int)
st.write("Dữ liệu MNIST đã được tải thành công!")

# Bước 2: Chia tách dữ liệu
st.header("2. Chia tách dữ liệu")
test_size = st.slider("Chọn tỷ lệ dữ liệu Test", 0.1, 0.5, 0.2, 0.05)
valid_size = st.slider("Chọn tỷ lệ dữ liệu Validation từ Train", 0.1, 0.3, 0.2, 0.05)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=valid_size, random_state=42)

st.write(f"Dữ liệu Train: {X_train.shape}, Validation: {X_valid.shape}, Test: {X_test.shape}")

# Bước 3: Huấn luyện và đánh giá mô hình
st.header("3. Huấn luyện và đánh giá mô hình")
model_choice = st.selectbox("Chọn mô hình", ["Decision Tree", "SVM"])

# Khởi tạo scaler và model lưu trong session
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None

if st.button("Huấn luyện mô hình"):
    with mlflow.start_run():
        if model_choice == "Decision Tree":
            model = DecisionTreeClassifier()
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)
            st.session_state.scaler = scaler
            model = SVC()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        accuracy = accuracy_score(y_valid, y_pred)

        mlflow.log_param("model", model_choice)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        st.session_state.model = model  # Lưu mô hình vào session

        st.write(f"Độ chính xác trên tập validation: {accuracy:.4f}")
        st.text(classification_report(y_valid, y_pred))

# Bước 4: Demo dự đoán
st.header("4. Demo dự đoán")
uploaded_file = st.file_uploader("Tải lên hình ảnh chữ số viết tay", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    from PIL import Image
    import cv2

    if st.session_state.model is None:
        st.error("Bạn cần huấn luyện mô hình trước khi dự đoán!")
    else:
        image = Image.open(uploaded_file).convert('L')
        image = image.resize((28, 28))
        img_array = np.array(image).reshape(1, -1) / 255.0

        if model_choice == "SVM" and st.session_state.scaler is not None:
            img_array = st.session_state.scaler.transform(img_array)
        prediction = st.session_state.model.predict(img_array)

        st.image(image, caption="Hình ảnh tải lên", use_column_width=True)
        st.write(f"Dự đoán: {prediction[0]}")
