import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Tiêu đề ứng dụng
st.title("Titanic Data Analysis & Preprocessing với Regression")
# Phần upload file
st.header("1. Dữ liệu")
df = pd.read_csv("titanic.csv")
st.write("### Dữ liệu gốc", df)

st.header("2. Tiền xử lý dữ liệu")
st.write("Chúng ta sẽ chọn dự đoán biến 'Fare' dựa trên các đặc trưng: 'Pclass', 'Age', 'SibSp', 'Parch'.")

# Kiểm tra các cột cần thiết có tồn tại không
required_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    st.error(f"Bộ dữ liệu thiếu các cột: {missing_cols}")
else:
    # Xử lý missing values: Điền giá trị trung vị cho Age
    df['Age'] = df['Age'].fillna(df['Age'].median())
    # Loại bỏ các dòng có missing target
    df = df.dropna(subset=['Fare'])
    
    # Chia dữ liệu
    X = df.drop("Survived", axis=1)  # Đặc trưng
    y = df["Survived"]               # Mục tiêu
    
    st.write("### Một số dòng dữ liệu của các feature", X.head())
    st.write("### Một số dòng dữ liệu của target", y.head())
    
    st.header("3. Chia tách dữ liệu")
    st.write("Nhập tỉ lệ (phần trăm) chia dữ liệu cho Train, Validation và Test (tổng phải = 100).")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        train_ratio = st.number_input("Train (%)", min_value=1, max_value=100, value=70)
    with col2:
        valid_ratio = st.number_input("Validation (%)", min_value=1, max_value=100, value=15)
    with col3:
        test_ratio = st.number_input("Test (%)", min_value=1, max_value=100, value=15)
        
    total = train_ratio + valid_ratio + test_ratio
    if total != 100:
        st.warning(f"Tổng tỉ lệ hiện tại là {total}, vui lòng đảm bảo tổng bằng 100.")
    else:
        # Tách dữ liệu: Đầu tiên tách ra Test, sau đó tách train & validation từ phần còn lại
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_ratio/100, random_state=42)
        valid_ratio_adjusted = valid_ratio / (train_ratio + valid_ratio)
        X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=valid_ratio_adjusted, random_state=42)
        
        st.write("Hình dạng của tập Train:", X_train.shape)
        st.write("Hình dạng của tập Validation:", X_valid.shape)
        st.write("Hình dạng của tập Test:", X_test.shape)
        
        # Gộp tập Train và Validation để thực hiện Cross Validation
        X_train_valid = pd.concat([X_train, X_valid])
        y_train_valid = pd.concat([y_train, y_valid])
        
        st.header("4. Huấn luyện & Kiểm thử mô hình")
        st.write("Chọn thuật toán huấn luyện:")
        algorithm = st.selectbox("Thuật toán:", ["Multiple Regression", "Polynomial Regression"])
        
        if algorithm == "Polynomial Regression":
            degree = st.number_input("Chọn bậc của đa thức:", min_value=2, max_value=5, value=2)
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('linear', LinearRegression())
            ])
        else:
            model = LinearRegression()
        
        st.subheader("Cross Validation (5-fold) trên tập Train+Validation")
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        # Sử dụng R2 score (có thể thay đổi scoring theo yêu cầu)
        scores = cross_val_score(model, X_train_valid, y_train_valid, cv=cv, scoring='r2')
        st.write("Điểm Cross Validation (R2):", scores)
        st.write("Điểm R2 trung bình:", np.mean(scores))
        
        # Huấn luyện trên tập Train+Validation và đánh giá trên tập Test
        model.fit(X_train_valid, y_train_valid)
        test_score = model.score(X_test, y_test)
        st.subheader("Đánh giá trên tập Test")
        st.write("Điểm R2 trên tập Test:", test_score)
        
        st.subheader("Biểu đồ Actual vs Predicted trên tập Test")
        y_pred = model.predict(X_test)
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color='blue', alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Giá trị thực (Fare)")
        ax.set_ylabel("Giá trị dự đoán (Fare)")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
