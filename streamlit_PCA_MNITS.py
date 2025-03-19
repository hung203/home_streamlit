import numpy as np
import pandas as pd
import streamlit as st
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')  # Đảm bảo không dùng GUI backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import datetime
import plotly.express as px

mlflow.set_tracking_uri("file:./mlruns")

# ------------------ KHAI BÁO BIẾN TRẠNG THÁI ------------------
if "mnist_loaded" not in st.session_state:
    st.session_state.mnist_loaded = False
if "X_pca" not in st.session_state:
    st.session_state.X_pca = None

# ------------------ HÀM TẢI DỮ LIỆU MNIST (CACHED) ------------------
@st.cache_data
def tai_du_lieu_MNIST(sample_size):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, cache=True)
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    total_samples = mnist.data.shape[0]
    if sample_size < total_samples:
        X, _, y, _ = train_test_split(
            X, y, train_size=sample_size, random_state=42, stratify=y
        )
    return X, y, total_samples

# ------------------ HÀM VẼ BIỂU ĐỒ TRỰC QUAN HÓA ------------------
@st.cache_data
def ve_bieu_do(X, y, title):
    df = pd.DataFrame({
        'X': X[:, 0],
        'Y': X[:, 1],
        'Label': y.astype(str)
    })
    fig = px.scatter(
        df,
        x='X', y='Y', color='Label', title=title,
        labels={'X': 'Thành phần 1', 'Y': 'Thành phần 2'},
        color_discrete_sequence=px.colors.qualitative.T10 if len(set(y)) <= 10 else px.colors.qualitative.Dark24,
        opacity=0.7, height=600
    )
    fig.update_layout(
        legend_title_text='Chữ số', title_font_size=14,
        xaxis_title_font_size=12, yaxis_title_font_size=12,
        legend=dict(x=1.05, y=1)
    )
    return fig

@st.cache_data
def ve_bieu_do_3d(X, y, title):
    df = pd.DataFrame({
        'X': X[:, 0], 'Y': X[:, 1], 'Z': X[:, 2],
        'Label': y.astype(str)
    })
    fig = px.scatter_3d(
        df,
        x='X', y='Y', z='Z', color='Label', title=title,
        labels={'X': 'Thành phần 1', 'Y': 'Thành phần 2', 'Z': 'Thành phần 3'},
        color_discrete_sequence=px.colors.qualitative.T10 if len(set(y)) <= 10 else px.colors.qualitative.Dark24,
        opacity=0.7, height=600
    )
    fig.update_layout(
        legend_title_text='Chữ số', title_font_size=14,
        scene=dict(
            xaxis_title='Thành phần 1', yaxis_title='Thành phần 2', zaxis_title='Thành phần 3',
            xaxis_title_font_size=12, yaxis_title_font_size=12, zaxis_title_font_size=12
        ),
        legend=dict(x=1.05, y=1)
    )
    return fig

# ------------------ TẠO 3 TAB ------------------
tab1, tab2, tab3 = st.tabs(["Lý thuyết về giảm chiều dữ liệu", "Thực hiện giảm chiều", "MLflow"])

# ----------- Tab 1: Lý thuyết về giảm chiều dữ liệu -----------
with tab1:
    st.header("Lý thuyết về giảm chiều dữ liệu")
    st.markdown(r"""
    ### Giới thiệu về giảm chiều dữ liệu 🚀
    **Giảm chiều dữ liệu** là quá trình chuyển đổi dữ liệu từ không gian có số chiều cao sang không gian có số chiều thấp hơn mà vẫn giữ lại được những đặc trưng quan trọng nhất. Quá trình này giúp:
    - **Trực quan hóa dữ liệu 📊:** Hiển thị dữ liệu trong 2D hoặc 3D.
    - **Giảm nhiễu và tăng tốc độ tính toán ⚡:** Loại bỏ đặc trưng dư thừa, tránh overfitting.
    """)

    st.header("📌 Lý thuyết về PCA", divider="blue")
    st.subheader("🔹 PCA là gì?")
    st.write("PCA chuyển đổi dữ liệu thành các thành phần chính giữ phần lớn phương sai.")
    
    st.subheader("2️⃣ Các bước thực hiện PCA", divider="blue")
    st.markdown("#### **Bước 1: Tìm vector trung bình**")
    st.write(r"Vector trung bình: $\mathbf{\mu} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i$")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("image/Screenshot 2025-03-12 203225.png")

    st.markdown("#### **Bước 2: Trừ vector trung bình**")
    st.write(r"Dữ liệu chuẩn hóa: $\hat{\mathbf{X}} = \mathbf{X} - \mathbf{\mu}$")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("image/Screenshot 2025-03-12 203258.png")

    st.markdown("#### **Bước 3: Tính ma trận hiệp phương sai**")
    st.write(r"$\mathbf{S} = \frac{1}{N} \hat{\mathbf{X}}^T \hat{\mathbf{X}}$")
    
    st.markdown("#### **Bước 4: Tính giá trị riêng và vector riêng**")
    st.write(r"Tìm $\lambda_i$ và $\mathbf{u}_i$ sao cho $\mathbf{S} \mathbf{u}_i = \lambda_i \mathbf{u}_i$")
    
    st.markdown("#### **Bước 5: Chọn $K$ vector riêng**")
    st.write(r"Chọn $K$ $\mathbf{u}_i$ có $\lambda_i$ lớn nhất.")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("image/Screenshot 2025-03-12 203337.png")

    st.markdown("#### **Bước 6: Chuyển đổi dữ liệu**")
    st.write(r"$\mathbf{Z} = \hat{\mathbf{X}} \mathbf{U}_K$, với $\mathbf{U}_K$ là ma trận $K$ vector riêng.")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("image/Screenshot 2025-03-12 203327.png")

    st.markdown("#### **Bước 7: Kết quả**")
    st.write(r"$\mathbf{Z}$ là dữ liệu trong không gian $K$ chiều.")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("image/Screenshot 2025-03-12 203306.png")

    st.header("📌 Lý thuyết về t-SNE", divider="blue")
    st.markdown("#### Bước 1: Tính \( p_{j|i} \)")
    st.write("Dùng Gaussian: \( p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma_i^2)} \)")
    st.image('https://miro.medium.com/v2/resize:fit:4800/format:webp/0*pTTqRArwYV_tGnF0.png')

    st.markdown("#### Bước 2: Khởi tạo ngẫu nhiên")
    st.write("Khởi tạo \( y_i \) trong không gian thấp.")
    st.image('https://miro.medium.com/v2/resize:fit:1100/format:webp/0*sNHrck20Xt7uS7X9.png')

    st.markdown("#### Bước 3: Tính \( q_{ij} \)")
    st.write(r"Dùng Student-t: \( q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||y_k - y_l||^2)^{-1}} \)")
    
    st.markdown("#### Bước 4: Tối ưu hóa")
    st.write(r"Giảm KL Divergence: \( KL(P||Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}} \)")
    st.image('https://miro.medium.com/v2/resize:fit:1100/format:webp/0*gx5m_CS7gVUn8WLH.gif')

    st.markdown("#### Bước 5: Kỹ thuật tối ưu")
    st.write("Dùng early exaggeration và compression.")
    
    st.markdown("#### Bước 6: Kết quả")
    st.write("Biểu diễn dữ liệu trong 2D/3D.")

# ----------- Tab 2: Thực hiện giảm chiều -----------
with tab2:
    st.title("Trực quan hóa PCA & t-SNE trên MNIST")
    
    sample_size = st.number_input(
        "Chọn số lượng mẫu dữ liệu sử dụng", min_value=1000, max_value=10000, value=5000, step=1000
    )
    if st.button("Tải dữ liệu MNIST"):
        X, y, total_samples = tai_du_lieu_MNIST(sample_size)
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.total_samples = total_samples
        st.session_state.mnist_loaded = True
        st.success(f"Dữ liệu MNIST đã được tải với {sample_size} mẫu!")

    if st.session_state.mnist_loaded:
        X = st.session_state.X
        y = st.session_state.y
        
        option = st.radio("Chọn thuật toán:", ("PCA", "t-SNE"))
        scaler = StandardScaler()
        
        @st.cache_data
        def standardize_data(X):
            return scaler.fit_transform(X)
        
        X_scaled = standardize_data(X)
        
        if option == "PCA":
            st.subheader("Cấu hình PCA")
            n_components = st.slider("Số thành phần", 2, 10, 2)  # Giới hạn để nhẹ hơn
            
            if st.button("Chạy PCA", key="btn_pca"):
                with st.spinner("Đang tính toán PCA..."):
                    pca = PCA(n_components=n_components)
                    X_pca = pca.fit_transform(X_scaled)
                    st.session_state.X_pca = X_pca
                    
                    explained_variance_ratio = pca.explained_variance_ratio_
                    total_variance = sum(explained_variance_ratio)
                
                st.success("PCA hoàn thành!")
                if n_components == 2:
                    fig_pca = ve_bieu_do(X_pca[:, :2], y, "Trực quan hóa PCA 2D")
                    st.plotly_chart(fig_pca, use_container_width=True)
                elif n_components == 3:
                    fig_pca = ve_bieu_do_3d(X_pca, y, "Trực quan hóa PCA 3D")
                    st.plotly_chart(fig_pca, use_container_width=True)
                else:
                    st.write("Tỷ lệ phương sai:", explained_variance_ratio)
                    st.write("Tổng phương sai:", total_variance)
                
                experiment_name = f"PCA_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
                mlflow.set_experiment(experiment_name)
                with mlflow.start_run():
                    mlflow.log_param("n_components", n_components)
                    mlflow.log_metric("total_explained_variance", total_variance)
        
        elif option == "t-SNE":
            st.subheader("Cấu hình t-SNE")
            n_components = st.slider("Số chiều", 2, 3, 2)  # Giới hạn để tránh nặng
            
            if st.button("Chạy t-SNE", key="btn_tsne"):
                with st.spinner("Đang tính toán t-SNE..."):
                    tsne = TSNE(n_components=n_components, method='barnes_hut', random_state=42)
                    X_tsne = tsne.fit_transform(X_scaled[:5000])  # Giới hạn mẫu để nhanh hơn
                
                st.success("t-SNE hoàn thành!")
                if n_components == 2:
                    fig_tsne = ve_bieu_do(X_tsne, y[:5000], "Trực quan hóa t-SNE 2D")
                    st.plotly_chart(fig_tsne, use_container_width=True)
                elif n_components == 3:
                    fig_tsne = ve_bieu_do_3d(X_tsne, y[:5000], "Trực quan hóa t-SNE 3D")
                    st.plotly_chart(fig_tsne, use_container_width=True)
                
                experiment_name = f"tSNE_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
                mlflow.set_experiment(experiment_name)
                with mlflow.start_run():
                    mlflow.log_param("n_components", n_components)
                    mlflow.log_metric("kl_divergence", tsne.kl_divergence_)

# ----------- Tab 3: MLflow -----------
with tab3:
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
