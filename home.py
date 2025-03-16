import importlib
import streamlit as st

# Khởi tạo session_state nếu chưa có
if 'last_option' not in st.session_state:
    st.session_state.last_option = None

# Tạo selectbox để chọn dự án
option = st.sidebar.selectbox(
    "📌 Chọn một dự án để thực hiện:",
    ["Phân tích Titanic", "Classification MNIST", "Clustering Algorithms MNIST", "PCA & t-SNE","Neural network"]
)

# Kiểm tra nếu option thay đổi
if st.session_state.last_option != option:
    # Xóa toàn bộ session state (trừ last_option)
    keys_to_remove = [key for key in st.session_state.keys() if key != 'last_option']
    for key in keys_to_remove:
        del st.session_state[key]
    
    # Cập nhật last_option
    st.session_state.last_option = option

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
elif option == "Neural network":
    with open("streamlit_neral.py", "r", encoding="utf-8") as file:
        code = file.read()
        exec(code)