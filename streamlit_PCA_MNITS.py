import pandas as pd
import streamlit as st
import mlflow
import mlflow.sklearn
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

# ------------------ HÀM TẢI DỮ LIỆU MNIST ------------------
def tai_du_lieu_MNIST():
    if "mnist_data" not in st.session_state:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        st.session_state.mnist_data = mnist
        st.session_state.total_samples = mnist.data.shape[0]
    
    sample_size = st.number_input(
        "Chọn số lượng mẫu dữ liệu sử dụng",
        min_value=1000,
        max_value=st.session_state.total_samples,
        value=st.session_state.total_samples,
        step=1000,
        help="Chọn số mẫu dữ liệu để giảm thời gian tính toán (mặc định là toàn bộ dữ liệu)"
    )
    
    if st.button("Tải dữ liệu MNIST"):
        mnist = st.session_state.mnist_data
        X, y = mnist.data / 255.0, mnist.target.astype(int)
        if sample_size < st.session_state.total_samples:
            X, _, y, _ = train_test_split(
                X, y, train_size=sample_size, random_state=42, stratify=y
            )
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.mnist_loaded = True
        st.success(f"Dữ liệu MNIST đã được tải với {sample_size} mẫu!")

# ------------------ HÀM VẼ BIỂU ĐỒ TRỰC QUAN HÓA ------------------
def ve_bieu_do(X, y, title):
    df = pd.DataFrame({
        'X': X[:, 0],
        'Y': X[:, 1],
        'Label': y.astype(str)
    })
    fig = px.scatter(
        df,
        x='X',
        y='Y',
        color='Label',
        title=title,
        labels={'X': 'Thành phần 1', 'Y': 'Thành phần 2'},
        color_discrete_sequence=px.colors.qualitative.T10 if len(set(y)) <= 10 else px.colors.qualitative.Dark24,
        opacity=0.7,
        height=600
    )
    fig.update_layout(
        legend_title_text='Chữ số',
        title_font_size=14,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12,
        legend=dict(x=1.05, y=1)
    )
    return fig

def ve_bieu_do_3d(X, y, title):
    df = pd.DataFrame({
        'X': X[:, 0],
        'Y': X[:, 1],
        'Z': X[:, 2],
        'Label': y.astype(str)
    })
    fig = px.scatter_3d(
        df,
        x='X',
        y='Y',
        z='Z',
        color='Label',
        title=title,
        labels={'X': 'Thành phần 1', 'Y': 'Thành phần 2', 'Z': 'Thành phần 3'},
        color_discrete_sequence=px.colors.qualitative.T10 if len(set(y)) <= 10 else px.colors.qualitative.Dark24,
        opacity=0.7,
        height=600
    )
    fig.update_layout(
        legend_title_text='Chữ số',
        title_font_size=14,
        scene=dict(
            xaxis_title='Thành phần 1',
            yaxis_title='Thành phần 2',
            zaxis_title='Thành phần 3',
            xaxis_title_font_size=12,
            yaxis_title_font_size=12,
            zaxis_title_font_size=12
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

    # Bước 1
    st.markdown("#### **Bước 1: Tìm vector trung bình (Find mean vector)**")
    st.write(r"""
    - Tính vector trung bình của dữ liệu ban đầu (ký hiệu là $\mathbf{X}$).  
    - Vector trung bình được biểu diễn bằng đường thẳng màu xanh lá cây chạy qua tâm của tập dữ liệu.
    """)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("image/Screenshot 2025-03-12 203225.png")

    # Bước 2
    st.markdown("#### **Bước 2: Trừ vector trung bình (Subtract mean)**")
    st.write(r"""
    - Trừ vector trung bình khỏi từng điểm dữ liệu để đưa dữ liệu về tâm tại gốc tọa độ (ký hiệu là $\hat{\mathbf{X}}$).  
    - Kết quả là dữ liệu đã được dịch chuyển, với trung bình bằng 0.
    """)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("image/Screenshot 2025-03-12 203258.png")

    # Bước 3
    st.markdown("#### **Bước 3: Tính ma trận hiệp phương sai (Compute covariance matrix)**")
    st.write(r"""
    - Tính ma trận hiệp phương sai $\mathbf{S}$ từ dữ liệu đã chuẩn hóa.  
    - Công thức: $\mathbf{S} = \frac{1}{N} \mathbf{X} \mathbf{X}^T$, trong đó $\mathbf{X}$ là ma trận dữ liệu, $N$ là số mẫu.
    """)

    # Bước 4
    st.markdown("#### **Bước 4: Tính giá trị riêng (eigenvalues) và vector riêng (eigenvectors) của $\mathbf{S}$**")
    st.write(r"""
    - Tìm các giá trị riêng ($\lambda_1, \lambda_2, \ldots, \lambda_D$) và vector riêng ($\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_D$) của ma trận hiệp phương sai $\mathbf{S}$.  
    - Vector riêng đại diện cho hướng của thành phần chính, giá trị riêng cho biết mức độ biến thiên.
    """)

    # Bước 5
    st.markdown(r"#### **Bước 5: Chọn $K$ vector riêng có giá trị riêng lớn nhất (Pick $K$ eigenvectors with highest eigenvalues)**")
    st.write(r"""
    - Chọn $K$ vector riêng tương ứng với $K$ giá trị riêng lớn nhất (ví dụ: $\mathbf{u}_1$ và $\mathbf{u}_2$).  
    - Các vector này được vẽ bằng đường đỏ, đại diện cho các thành phần chính.
    """)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("image/Screenshot 2025-03-12 203337.png")

    # Bước 6
    st.markdown("#### **Bước 6: Chuyển đổi dữ liệu sang vector riêng đã chọn (Project data to selected eigenvectors)**")
    st.write(r"""
    - Nhân dữ liệu đã chuẩn hóa ($\hat{\mathbf{X}}$) với ma trận gồm $K$ vector riêng để thu dữ liệu trong không gian mới.  
    - Dữ liệu được chiếu lên các hướng $\mathbf{u}_1$ và $\mathbf{u}_2$.
    """)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("image/Screenshot 2025-03-12 203327.png")

    # Bước 7
    st.markdown("#### **Bước 7: Lấy các điểm chiếu trong không gian chiều thấp (Obtain projected points in low dimension)**")
    st.write(r"""
    - Dữ liệu cuối cùng được biểu diễn trong không gian $K$ chiều (ví dụ: 2D).  
    - Kết quả là tập dữ liệu mới $\mathbf{Z}$, giữ lại phần lớn thông tin từ dữ liệu gốc.
    """)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("image/Screenshot 2025-03-12 203306.png")

    # Lưu ý quan trọng về PCA
    st.markdown("### 🎨 Lưu ý quan trọng về PCA")
    st.write(r"""
    - Hình ảnh minh họa nhấn mạnh rằng PCA giảm chiều dữ liệu bằng cách giữ các thành phần chính có phương sai lớn nhất.  
    - Các bước trên giả định dữ liệu đã được chuẩn hóa (trung bình = 0, độ lệch chuẩn = 1).
    """)

      
    # ### Ứng dụng của giảm chiều dữ liệu 💼

    # # - **Trực quan hóa dữ liệu:**  
    # #   Giúp các nhà khoa học dữ liệu và kỹ sư hiểu được cấu trúc nội tại của dữ liệu, nhận diện các mẫu bất thường và phân nhóm dữ liệu.
    # # - **Tiền xử lý cho học máy:**  
    # #   Giảm số chiều dữ liệu giúp giảm độ phức tạp của mô hình, tăng hiệu suất tính toán và giảm nguy cơ quá khớp.
    # # - **Khai phá dữ liệu:**  
    # #   Phát hiện các mối quan hệ ẩn, hiểu sâu hơn về dữ liệu và đưa ra các quyết định kinh doanh dựa trên dữ liệu.
    
    # # ### Lưu ý khi thực hiện giảm chiều dữ liệu ⚠️
    # # - **Lựa chọn thuật toán:**  
    # #   Tùy vào đặc điểm của dữ liệu và mục tiêu phân tích mà bạn có thể lựa chọn phương pháp giảm chiều phù hợp (PCA cho dữ liệu tuyến tính, t-SNE cho dữ liệu phi tuyến).
    # # - **Tinh chỉnh tham số:**  
    # #   Các tham số như số lượng thành phần trong PCA, perplexity và learning rate trong t-SNE rất quan trọng và cần được thử nghiệm để đạt được kết quả tốt nhất.
    # # - **Hiểu rõ dữ liệu:**  
    # #   Phân tích và hiểu rõ dữ liệu ban đầu sẽ giúp việc lựa chọn phương pháp và cấu hình tham số trở nên hiệu quả hơn.
    # """)

# ----------- Tab 2: Thực hiện giảm chiều -----------
with tab2:
    st.title("Trực quan hóa PCA & t-SNE trên MNIST")
    if not st.session_state.mnist_loaded:
        tai_du_lieu_MNIST()
    
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
                help="Số thành phần chính cần giữ lại. Nếu > 3, hiển thị tỷ lệ phương sai thay vì hình ảnh."
            )
            
            experiment_name_pca = st.text_input(
                "Đặt tên cho thí nghiệm MLflow (PCA)",
                value="",
                help="Nhập tên thí nghiệm cho MLflow. Nếu để trống, hệ thống sẽ tự động tạo tên dựa trên timestamp."
            )
            
            if st.button("Chạy PCA", key="btn_pca"):
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                if experiment_name_pca.strip() == "":
                    experiment_name = f"Experiment_PCA_{timestamp}"
                else:
                    experiment_name = experiment_name_pca.strip()
                
                mlflow.set_experiment(experiment_name)
                st.session_state.experiment_name = experiment_name
                st.write("Tên thí nghiệm:", experiment_name)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Đang tính toán PCA..."):
                    status_text.text("Đang chuẩn hóa dữ liệu...")
                    progress_bar.progress(10)
                    
                    status_text.text("Đang thực hiện PCA...")
                    pca = PCA(n_components=n_components)
                    X_pca = pca.fit_transform(X_scaled)
                    st.session_state.X_pca = X_pca
                    
                    # Tính total_variance trong mọi trường hợp
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
                    fig_pca = ve_bieu_do_3d(X_pca, y, "Trực quan hóa PCA 3D")
                    st.plotly_chart(fig_pca, use_container_width=True, renderer="plotly_mimetype")
                else:
                    st.write("**Tỷ lệ phương sai giải thích cho từng chiều:**", explained_variance_ratio)
                    st.write("**Tổng tỷ lệ phương sai giữ lại:**", total_variance)
                
                with mlflow.start_run():
                    mlflow.log_param("n_components", n_components)
                    mlflow.log_metric("total_explained_variance", total_variance)
                st.success("Kết quả PCA đã được lưu với MLflow!")
        
        elif option == "t-SNE":
            st.subheader("Cấu hình t-SNE")
            n_components = st.slider(
                "Chọn số chiều đầu ra (n_components)",
                2, 50, 2,
                help="Số chiều để giảm. Nếu > 3, dùng thuật toán 'exact' và hiển thị KL Divergence."
            )
            
            experiment_name_tsne = st.text_input(
                "Đặt tên cho thí nghiệm MLflow (t-SNE)",
                value="",
                help="Nhập tên thí nghiệm cho MLflow. Nếu để trống, hệ thống sẽ tự động tạo tên dựa trên timestamp."
            )
            
            if st.button("Chạy t-SNE", key="btn_tsne"):
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                if experiment_name_tsne.strip() == "":
                    experiment_name = f"Experiment_tSNE_{timestamp}"
                else:
                    experiment_name = experiment_name_tsne.strip()
                
                mlflow.set_experiment(experiment_name)
                st.session_state.experiment_name = experiment_name
                st.write("Tên thí nghiệm:", experiment_name)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Đang tính toán t-SNE..."):
                    status_text.text("Đang chuẩn hóa dữ liệu...")
                    progress_bar.progress(10)
                    
                    if n_components <= 3:
                        method = 'barnes_hut'
                    else:
                        method = 'exact'
                    
                    status_text.text("Đang thực hiện t-SNE... (có thể mất vài phút)")
                    tsne = TSNE(n_components=n_components, method=method, random_state=42)
                    
                    progress_bar.progress(50)
                    X_tsne = tsne.fit_transform(X_scaled)
                    
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
                    st.info("KL Divergence càng nhỏ thì cấu trúc cục bộ của dữ liệu càng được bảo toàn tốt.")
                
                with mlflow.start_run():
                    mlflow.log_param("n_components", n_components)
                    mlflow.log_param("method", method)
                    if n_components > 3:  # Chỉ ghi kl_divergence khi n_components > 3
                        mlflow.log_metric("kl_divergence", kl_divergence)
                st.success("Kết quả t-SNE đã được lưu với MLflow!")

# ----------- Tab 3: MLflow -----------
with tab3:
    st.header("Tracking MLflow")
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
    
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
    
            selected_exp_name = st.selectbox(
                "🔍 Chọn thí nghiệm để xem chi tiết",
                options=[exp.name for exp in experiments]
            )
    
            selected_exp_id = next(exp.experiment_id for exp in experiments if exp.name == selected_exp_name)
    
            runs = mlflow.search_runs(selected_exp_id)
            if not runs.empty:
                st.write("#### Danh sách runs")
                st.dataframe(runs)
    
                selected_run_id = st.selectbox(
                    "🔍 Chọn run để xem chi tiết",
                    options=runs["run_id"]
                )
    
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