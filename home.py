import importlib
import streamlit as st

# Tạo selectbox để chọn dự án
option = st.sidebar.selectbox(
    "📌 Chọn một dự án để thực hiện:",
    ["Phân tích Titanic", "Classification MNIST", "Clustering Algorithms MNIST", "PCA & t-SNE"]
)

# Hiển thị nội dung tương ứng với lựa chọn
if option == "Phân tích Titanic":
    with open("streamlit_titanic.py", "r", encoding="utf-8") as file:
        code = file.read()
        exec(code)
elif option == "Classification MNIST":
    with open("streamlit_Mnits.py", "r", encoding="utf-8") as file:
        code = file.read()
        exec(code)
elif option == "Clustering Algorithms MNIST":
    with open("streamlit_cluter.py", "r", encoding="utf-8") as file:
        code = file.read()
        exec(code)
elif option == "PCA & t-SNE":
    with open("streamlit_PCA_MNITS.py", "r", encoding="utf-8") as file:
        code = file.read()
        exec(code)
