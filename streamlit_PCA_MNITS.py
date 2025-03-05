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
def ve_bieu_do(X_embedded, y, tieu_de):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='tab10', alpha=0.6)
    ax.set_title(tieu_de)
    ax.set_xlabel("Thành phần 1")
    ax.set_ylabel("Thành phần 2")
    plt.colorbar(scatter, ax=ax, label="Chữ số")
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

    ### Các phương pháp giảm chiều dữ liệu phổ biến 🔍

    #### 1. PCA (Principal Component Analysis) 💡
    - **Nguyên lý:**  
      PCA tìm các thành phần chính sao cho phần lớn phương sai của dữ liệu được giữ lại. Giả sử dữ liệu đã được trung bình hóa, ta có:
      
      - **Ma trận hiệp phương sai:**
      $$ \Sigma = \frac{1}{n-1} X^T X $$
      
      - **Phân tích giá trị riêng:**
      $$ \Sigma v = \lambda v $$
      
      - **Chiếu dữ liệu lên không gian các thành phần chính:**
      $$ Z = XW $$
      
      Trong đó, **W** là ma trận chứa các vector riêng (eigenvectors) tương ứng với các giá trị riêng (eigenvalues) lớn nhất.

    #### 2. t-SNE (t-distributed Stochastic Neighbor Embedding) 🔥
    - **Nguyên lý:**  
      t-SNE trực quan hóa dữ liệu bằng cách chuyển đổi khoảng cách giữa các điểm trong không gian cao chiều thành xác suất, sau đó tái hiện các mối quan hệ này trong không gian 2D hoặc 3D:
      
      - **Xác suất khoảng cách trong không gian cao chiều:**
      $$ p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)} $$
      
      - **Xác suất đối xứng:**
      $$ p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n} $$
      
      - **Trong không gian thấp chiều, sử dụng phân phối Student’s t:**
      $$ q_{ij} = \frac{(1+\|y_i-y_j\|^2)^{-1}}{\sum_{k \neq l}(1+\|y_k-y_l\|^2)^{-1}} $$
      
      - **Hàm mất mát Kullback-Leibler cần tối thiểu hóa:**
      $$ KL(P\|Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}} $$
      
    ### Ứng dụng của giảm chiều dữ liệu 💼
    - **Trực quan hóa dữ liệu:**  
      Giúp các nhà khoa học dữ liệu và kỹ sư hiểu được cấu trúc nội tại của dữ liệu, nhận diện các mẫu bất thường và phân nhóm dữ liệu.
    - **Tiền xử lý cho học máy:**  
      Giảm số chiều dữ liệu giúp giảm độ phức tạp của mô hình, tăng hiệu suất tính toán và giảm nguy cơ quá khớp.
    - **Khai phá dữ liệu:**  
      Phát hiện các mối quan hệ ẩn, hiểu sâu hơn về dữ liệu và đưa ra các quyết định kinh doanh dựa trên dữ liệu.
    
    ### Lưu ý khi thực hiện giảm chiều dữ liệu ⚠️
    - **Lựa chọn thuật toán:**  
      Tùy vào đặc điểm của dữ liệu và mục tiêu phân tích mà bạn có thể lựa chọn phương pháp giảm chiều phù hợp (PCA cho dữ liệu tuyến tính, t-SNE cho dữ liệu phi tuyến).
    - **Tinh chỉnh tham số:**  
      Các tham số như số lượng thành phần trong PCA, perplexity và learning rate trong t-SNE rất quan trọng và cần được thử nghiệm để đạt được kết quả tốt nhất.
    - **Hiểu rõ dữ liệu:**  
      Phân tích và hiểu rõ dữ liệu ban đầu sẽ giúp việc lựa chọn phương pháp và cấu hình tham số trở nên hiệu quả hơn.
    """)

# ----------- Tab 2: Thực hiện giảm chiều -----------
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import datetime
import mlflow
import pandas as pd

# Hàm vẽ biểu đồ 2D (Matplotlib)
def ve_bieu_do(X, y, title):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=10)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    return fig

# Hàm vẽ biểu đồ 3D tương tác (Plotly)
def ve_bieu_do_3d(X, y, title):
    df = pd.DataFrame({
        'X': X[:, 0],
        'Y': X[:, 1],
        'Z': X[:, 2],
        'Label': y
    })
    fig = px.scatter_3d(
        df, 
        x='X', 
        y='Y', 
        z='Z', 
        color='Label', 
        title=title,
        color_continuous_scale='Viridis',
        opacity=0.7,
        height=600
    )
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    return fig

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
            
            if st.button("Chạy PCA", key="btn_pca"):
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                experiment_name = f"Experiment_PCA_{timestamp}"
                mlflow.set_experiment(experiment_name)
                st.session_state.experiment_name = experiment_name
                st.write("Tên thí nghiệm:", experiment_name)
                
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_scaled)
                st.session_state.X_pca = X_pca
                st.success("PCA đã được tính!")
                
                st.subheader("Kết quả PCA")
                if n_components == 2:
                    fig_pca = ve_bieu_do(X_pca[:, :2], y, "Trực quan hóa PCA 2D")
                    st.pyplot(fig_pca)
                elif n_components == 3:
                    fig_pca = ve_bieu_do_3d(X_pca, y, "Trực quan hóa PCA 3D")
                    st.plotly_chart(fig_pca, use_container_width=True)
                else:
                    explained_variance_ratio = pca.explained_variance_ratio_
                    total_variance = sum(explained_variance_ratio)
                    st.write("**Tỷ lệ phương sai giải thích cho từng chiều:**", explained_variance_ratio)
                    st.write("**Tổng tỷ lệ phương sai giữ lại:**", total_variance)
                
                with mlflow.start_run():
                    mlflow.log_param("n_components", n_components)
                    if n_components == 2:
                        fig_pca.savefig("pca_visualization.png")
                        mlflow.log_artifact("pca_visualization.png")
                    elif n_components == 3:
                        fig_pca.write_image("pca_visualization.png")
                        mlflow.log_artifact("pca_visualization.png")
                    else:
                        mlflow.log_metric("total_explained_variance", total_variance)
                st.success("Kết quả PCA đã được lưu với MLflow!")
        
        elif option == "t-SNE":
            st.subheader("Cấu hình t-SNE")
            n_components = st.slider(
                "Chọn số chiều đầu ra (n_components)",
                2, 50, 2,
                help="Số chiều để giảm. Nếu > 3, dùng thuật toán 'exact' và hiển thị KL Divergence."
            )
            
            if st.button("Chạy t-SNE", key="btn_tsne"):
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                experiment_name = f"Experiment_tSNE_{timestamp}"
                mlflow.set_experiment(experiment_name)
                st.session_state.experiment_name = experiment_name
                st.write("Tên thí nghiệm:", experiment_name)
                
                # Chạy t-SNE trực tiếp trên X_scaled
                if n_components <= 3:
                    method = 'barnes_hut'
                else:
                    method = 'exact'               
                tsne = TSNE(n_components=n_components, method=method, random_state=42)
                X_tsne = tsne.fit_transform(X_scaled)  # Dùng X_scaled thay vì X_pca
                st.success("t-SNE đã được tính!")
                
                st.subheader("Kết quả t-SNE")
                if n_components == 2:
                    fig_tsne = ve_bieu_do(X_tsne, y, "Trực quan hóa t-SNE 2D")
                    st.pyplot(fig_tsne)
                elif n_components == 3:
                    fig_tsne = ve_bieu_do_3d(X_tsne, y, "Trực quan hóa t-SNE 3D")
                    st.plotly_chart(fig_tsne, use_container_width=True)
                else:
                    kl_divergence = tsne.kl_divergence_
                    st.write("**Giá trị KL Divergence:**", kl_divergence)
                    st.info("KL Divergence càng nhỏ thì cấu trúc cục bộ của dữ liệu càng được bảo toàn tốt.")
                
                with mlflow.start_run():
                    mlflow.log_param("n_components", n_components)
                    mlflow.log_param("method", method)
                    if n_components == 2:
                        fig_tsne.savefig("tsne_visualization.png")
                        mlflow.log_artifact("tsne_visualization.png")
                    elif n_components == 3:
                        fig_tsne.write_image("tsne_visualization.png")
                        mlflow.log_artifact("tsne_visualization.png")
                    else:
                        mlflow.log_metric("kl_divergence", kl_divergence)
                st.success("Kết quả t-SNE đã được lưu với MLflow!")
# ----------- Tab 3: MLflow -----------
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
    
            # Chọn thí nghiệm dựa trên tên
            selected_exp_name = st.selectbox(
                "🔍 Chọn thí nghiệm để xem chi tiết",
                options=[exp.name for exp in experiments]
            )
    
            # Lấy ID của thí nghiệm được chọn
            selected_exp_id = next(exp.experiment_id for exp in experiments if exp.name == selected_exp_name)
    
            # Lấy danh sách runs trong thí nghiệm được chọn
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
    
                st.write("##### Metrics")
                st.json(run.data.metrics)
    
                st.write("##### Params")
                st.json(run.data.params)
    
                # Liệt kê artifacts
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
