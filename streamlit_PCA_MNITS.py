import numpy as np
import pandas as pd
import streamlit as st
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')
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
if "X_tsne" not in st.session_state:
    st.session_state.X_tsne = None

# ------------------ HÀM TẢI DỮ LIỆU MNIST (CACHE) ------------------
@st.cache_data
def tai_du_lieu_MNIST(sample_size):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    total_samples = mnist.data.shape[0]
    if sample_size < total_samples:
        X, _, y, _ = train_test_split(
            X, y, train_size=sample_size / total_samples, random_state=42, stratify=y
        )
    return X, y, total_samples

# ------------------ HÀM VẼ BIỂU ĐỒ TRỰC QUAN HÓA ------------------
def ve_bieu_do(X, y, title):
    df = pd.DataFrame({
        'X': X[:, 0],
        'Y': X[:, 1],
        'Label': y.astype(str)
    })
    fig = px.scatter(
        df, x='X', y='Y', color='Label', title=title,
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

def ve_bieu_do_3d(X, y, title):
    df = pd.DataFrame({
        'X': X[:, 0], 'Y': X[:, 1], 'Z': X[:, 2],
        'Label': y.astype(str)
    })
    fig = px.scatter_3d(
        df, x='X', y='Y', z='Z', color='Label', title=title,
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
    **Giảm chiều dữ liệu** là quá trình chuyển đổi dữ liệu từ không gian có số chiều cao (với nhiều đặc trưng) sang không gian có số chiều thấp hơn mà vẫn giữ lại được những đặc trưng quan trọng nhất. Quá trình này giúp:
    - **Trực quan hóa dữ liệu 📊:** Hiển thị dữ liệu trong không gian 2D hoặc 3D, từ đó dễ dàng nhận biết cấu trúc, nhóm (clusters) hay các mối liên hệ giữa các mẫu.
    - **Giảm nhiễu và tăng tốc độ tính toán ⚡:** Loại bỏ những đặc trưng dư thừa, không cần thiết giúp mô hình học máy chạy nhanh hơn và tránh tình trạng quá khớp (overfitting).
    """)

    st.header("📌 Lý thuyết về PCA (Phân tích thành phần chính)", divider="blue")
    st.subheader("🔹 PCA là gì?")
    st.write(r"""
    PCA (Principal Component Analysis) là một kỹ thuật giảm chiều dữ liệu, chuyển đổi dữ liệu ban đầu thành các **thành phần chính** giữ phần lớn phương sai. Hình ảnh dưới đây minh họa các bước thực hiện PCA.
    """)

    st.subheader("2️⃣ Các bước thực hiện PCA", divider="blue")
    st.markdown("#### **Bước 1: Tìm vector trung bình (Find mean vector)**")
    st.write(r" - Tính vector trung bình của dữ liệu ban đầu (ký hiệu là $\mathbf{X}$).")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("image/Screenshot 2025-03-12 203225.png")

    st.markdown("#### **Bước 2: Trừ vector trung bình (Subtract mean)**")
    st.write(r" - Trừ vector trung bình khỏi từng điểm dữ liệu để đưa dữ liệu về tâm tại gốc tọa độ (ký hiệu là $\hat{\mathbf{X}}$).")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("image/Screenshot 2025-03-12 203258.png")

    st.markdown("#### **Bước 3: Tính ma trận hiệp phương sai (Compute covariance matrix)**")
    st.write(r" - Tính ma trận hiệp phương sai $\mathbf{S}$: $\mathbf{S} = \frac{1}{N} \mathbf{X} \mathbf{X}^T$.")

    st.markdown("#### **Bước 4: Tính giá trị riêng (eigenvalues) và vector riêng (eigenvectors) của $\mathbf{S}$**")
    st.write(r" - Tìm các giá trị riêng ($\lambda_1, \lambda_2, \ldots$) và vector riêng ($\mathbf{u}_1, \mathbf{u}_2, \ldots$).")

    st.markdown(r"#### **Bước 5: Chọn $K$ vector riêng có giá trị riêng lớn nhất**")
    st.write(r" - Chọn $K$ vector riêng với $K$ giá trị riêng lớn nhất.")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("image/Screenshot 2025-03-12 203337.png")

    st.markdown("#### **Bước 6: Chuyển đổi dữ liệu sang vector riêng đã chọn**")
    st.write(r" - Nhân $\hat{\mathbf{X}}$ với ma trận $K$ vector riêng.")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("image/Screenshot 2025-03-12 203327.png")

    st.markdown("#### **Bước 7: Lấy các điểm chiếu trong không gian chiều thấp**")
    st.write(r" - Thu dữ liệu mới $\mathbf{Z}$ trong không gian $K$ chiều.")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("image/Screenshot 2025-03-12 203306.png")

    st.markdown("### 🎨 Lưu ý quan trọng về PCA")
    st.write(r" - PCA giữ các thành phần chính có phương sai lớn nhất, dữ liệu cần được chuẩn hóa trước.")

    st.header("📌 Lý thuyết về t-SNE (t-Distributed Stochastic Neighbor Embedding)", divider="blue")
    st.markdown("#### Bước 1: Tính \( p_{j|i} \) trong không gian chiều cao bằng Gaussian và chuẩn hóa")
    st.write(" - Đo độ tương tự bằng phân phối Gaussian, chuẩn hóa tổng xác suất = 1.")
    st.image('https://miro.medium.com/v2/resize:fit:4800/format:webp/0*pTTqRArwYV_tGnF0.png')

    st.markdown("#### Bước 2: Khởi tạo ngẫu nhiên các điểm trong không gian chiều thấp")
    st.write(" - Tạo các điểm ngẫu nhiên trong không gian 2D/3D.")
    st.image('https://miro.medium.com/v2/resize:fit:1100/format:webp/0*sNHrck20Xt7uS7X9.png')

    st.markdown("#### Bước 3: Tính \( q_{ij} \) trong không gian chiều thấp bằng phân phối Student-t")
    st.write(" - Đo độ tương tự bằng phân phối Student-t.")

    st.markdown("#### Bước 4: Tối ưu hóa vị trí các điểm bằng gradient descent để giảm KL divergence")
    st.write(" - Điều chỉnh để \( q_{ij} \) giống \( p_{j|i} \) nhất.")
    st.image('https://miro.medium.com/v2/resize:fit:1100/format:webp/0*gx5m_CS7gVUn8WLH.gif')

    st.markdown("#### Bước 5: Áp dụng các kỹ thuật tối ưu (early exaggeration, early compression)")
    st.write(" - Cải thiện phân tách cụm và tránh chồng chéo.")

    st.markdown("#### Bước 6: Trả về biểu diễn trực quan cuối cùng")
    st.write(" - Thu bản đồ 2D/3D hiển thị cấu trúc cục bộ.")

    st.markdown("**Lưu ý**: t-SNE tập trung vào bảo tồn cấu trúc cục bộ, không phải khoảng cách toàn cục.")

# ----------- Tab 2: Thực hiện giảm chiều -----------
with tab2:
    st.title("Trực quan hóa PCA & t-SNE trên MNIST")
    
    # Fragment cho tải dữ liệu
    @st.fragment
    def load_data_interface():
        if not st.session_state.mnist_loaded:
            sample_size = st.number_input(
                "Chọn số lượng mẫu dữ liệu sử dụng",
                min_value=1000,
                max_value=70000,
                value=10000,
                step=1000,
                help="Chọn số mẫu dữ liệu để giảm thời gian tính toán (mặc định 10,000)"
            )
            if st.button("Tải dữ liệu MNIST"):
                X, y, total_samples = tai_du_lieu_MNIST(sample_size)
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.total_samples = total_samples
                st.session_state.mnist_loaded = True
                st.success(f"Dữ liệu MNIST đã được tải với {sample_size} mẫu!")
    
    load_data_interface()
    
    if st.session_state.mnist_loaded:
        X = st.session_state.X
        y = st.session_state.y
        st.write("Dữ liệu đã được tải thành công!")
        
        option = st.radio(
            "Chọn thuật toán cần chạy:",
            ("PCA", "t-SNE"),
            help="Chọn PCA để thu gọn dữ liệu hoặc t-SNE để trực quan hóa không gian dữ liệu."
        )
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if option == "PCA":
            st.subheader("Cấu hình PCA")
            n_components = st.slider(
                "Chọn số thành phần (n_components)",
                2, 50, 2,
                help="Số thành phần chính cần giữ lại."
            )
            
            experiment_name_pca = st.text_input(
                "Đặt tên cho thí nghiệm MLflow (PCA)",
                value=f"Experiment_PCA_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            
            # Fragment cho chạy PCA
            @st.fragment
            def run_pca():
                if st.button("Chạy PCA", key="btn_pca"):
                    mlflow.set_experiment(experiment_name_pca)
                    st.session_state.experiment_name = experiment_name_pca
                    st.write("Tên thí nghiệm:", experiment_name_pca)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with st.spinner("Đang tính toán PCA..."):
                        status_text.text("Đang chuẩn hóa dữ liệu...")
                        progress_bar.progress(10)
                        
                        status_text.text("Đang thực hiện PCA...")
                        pca = PCA(n_components=n_components)
                        X_pca = pca.fit_transform(X_scaled)
                        st.session_state.X_pca = X_pca
                        
                        explained_variance_ratio = pca.explained_variance_ratio_
                        total_variance = sum(explained_variance_ratio)
                        
                        progress_bar.progress(100)
                        status_text.text("Hoàn thành PCA!")
                    
                    st.success("PCA đã được tính!")
                    
                    st.subheader("Kết quả PCA")
                    if n_components == 2:
                        fig_pca = ve_bieu_do(X_pca[:, :2], y, "Trực quan hóa PCA 2D")
                        st.plotly_chart(fig_pca, use_container_width=True, renderer="plotly_mimetype")
                    elif n_components == 3:
                        fig_pca = ve_bieu_do_3d(X_pca, yPrerun(fig_pca)
                        st.plotly_chart(fig_pca, use_container_width=True, renderer="plotly_mimetype")
                    else:
                        st.write("**Tỷ lệ phương sai giải thích cho từng chiều:**", explained_variance_ratio)
                        st.write("**Tổng tỷ lệ phương sai giữ lại:**", total_variance)
                    
                    with mlflow.start_run():
                        mlflow.log_param("n_components", n_components)
                        mlflow.log_metric("total_explained_variance", total_variance)
                    st.success("Kết quả PCA đã được lưu với MLflow!")
            
            run_pca()
        
        elif option == "t-SNE":
            st.subheader("Cấu hình t-SNE")
            n_components = st.slider(
                "Chọn số chiều đầu ra (n_components)",
                2, 50, 2,
                help="Số chiều để giảm."
            )
            
            experiment_name_tsne = st.text_input(
                "Đặt tên cho thí nghiệm MLflow (t-SNE)",
                value=f"Experiment_tSNE_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            
            # Fragment cho chạy t-SNE
            @st.fragment
            def run_tsne():
                if st.button("Chạy t-SNE", key="btn_tsne"):
                    mlflow.set_experiment(experiment_name_tsne)
                    st.session_state.experiment_name = experiment_name_tsne
                    st.write("Tên thí nghiệm:", experiment_name_tsne)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with st.spinner("Đang tính toán t-SNE..."):
                        status_text.text("Đang chuẩn hóa dữ liệu...")
                        progress_bar.progress(10)
                        
                        method = 'barnes_hut' if n_components <= 3 else 'exact'
                        status_text.text("Đang thực hiện t-SNE... (có thể mất vài phút)")
                        tsne = TSNE(n_components=n_components, method=method, random_state=42)
                        progress_bar.progress(50)
                        X_tsne = tsne.fit_transform(X_scaled)
                        st.session_state.X_tsne = X_tsne
                        
                        progress_bar.progress(100)
                        status_text.text("Hoàn thành t-SNE!")
                    
                    st.success("t-SNE đã được tính!")
                    
                    st.subheader("Kết quả t-SNE")
                    if n_components == 2:
                        fig_tsne = ve_bieu_do(X_tsne, y, "Trực quan hóa t-SNE 2D")
                        st.plotly_chart(fig_tsne, use_container_width=True, renderer="plotly_mimetype")
                    elif n_components == 3:
                        fig_tsne = ve_bieu_do_3d(X_tsne, y, "Trực quan hóa t-SNE 3D")
                        st.plotly_chart(fig_tsne, use_container_width=True, renderer="plotly_mimetype")
                    else:
                        kl_divergence = tsne.kl_divergence_
                        st.write("**Giá trị KL Divergence:**", kl_divergence)
                        st.info("KL Divergence càng nhỏ thì cấu trúc cục bộ càng được bảo toàn tốt.")
                    
                    with mlflow.start_run():
                        mlflow.log_param("n_components", n_components)
                        mlflow.log_param("method", method)
                        if n_components > 3:
                            mlflow.log_metric("kl_divergence", kl_divergence)
                    st.success("Kết quả t-SNE đã được lưu với MLflow!")
            
            run_tsne()

# ----------- Tab 3: MLflow -----------
with tab3:
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
