import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Tiêu đề ứng dụng
st.title("Tiền xử lý dữ liệu Titanic cho Multiple Regression")

# Tải dữ liệu Titanic
st.header("1. Tải dữ liệu Titanic")
df = pd.read_csv("titanic.csv")
st.write("Dữ liệu ban đầu:")
st.write(df.head())

# Xử lý giá trị thiếu
st.header("2. Xử lý giá trị thiếu")

# Tính số lượng giá trị thiếu cho mỗi cột và chuyển thành DataFrame
missing_data = df.isnull().sum().reset_index()
missing_data.columns = ['Column', 'Missing Count']
st.write(missing_data)

st.write("### Điền giá trị thiếu cho cột Age, Fare, và Embarked:")

st.write("Đối với cột Age thay thế các giá trị thiếu (NaN) trong cột Age bằng giá trị trung vị vừa tính được.")
st.write("Đối với cột Fare thay thế các giá trị thiếu trong cột Fare bằng giá trị trung bình.")
st.write("Đối với cột Embarked thay thế các giá trị thiếu trong cột Embarked bằng giá trị mode vừa lấy được.")

# Lọc chỉ những cột có giá trị thiếu
missing_data = missing_data[missing_data['Missing Count'] > 0]
# Điền giá trị thiếu
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Fare"].fillna(df["Fare"].mean(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Xóa cột Cabin
df.drop("Cabin", axis=1, inplace=True)
st.write("#### Dữ liệu sau khi xử lý giá trị thiếu:")
st.write(df.head())

# Mã hóa dữ liệu
st.header("3. Mã hóa dữ liệu")
st.write("Mã hóa cột Sex và one-hot encoding cho Embarked và Title:")

# Mã hóa cột Sex
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# One-hot encoding cho Embarked và Title
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

st.write("Dữ liệu sau khi mã hóa:")
st.write(df.head())

# Xóa các cột không cần thiết
st.header("4. Xóa các cột không cần thiết")
st.write("Xóa các cột PassengerId, Name, và Ticket:")

# Xóa các cột không cần thiết
df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

st.write("Dữ liệu sau khi xóa các cột không cần thiết:")
st.write(df.head())

# Chuẩn hóa dữ liệu
st.header("5. Chuẩn hóa dữ liệu")
st.write("Chuẩn hóa các cột số (Age, Fare, SibSp, Parch:")

# Chuẩn hóa các cột số
scaler = StandardScaler()
numerical_features = ["Age", "Fare", "SibSp", "Parch"]
df[numerical_features] = scaler.fit_transform(df[numerical_features])

st.write("Dữ liệu sau khi chuẩn hóa:")
st.write(df.head())

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
st.header("6. Chia dữ liệu thành tập huấn luyện và tập kiểm tra")
st.write("Chia dữ liệu thành X (đặc trưng) và y (mục tiêu):")

# Chia dữ liệu
X = df.drop("Survived", axis=1)  # Đặc trưng
y = df["Survived"]               # Mục tiêu

# Chia thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write("Kích thước tập huấn luyện và tập kiểm tra:")
st.write(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
st.write(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Kết thúc
st.header("Hoàn thành tiền xử lý dữ liệu!")
st.write("Dữ liệu đã sẵn sàng để huấn luyện mô hình Multiple Regression.")