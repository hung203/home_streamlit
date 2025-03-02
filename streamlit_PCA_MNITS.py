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
    st.markdown("""
    ### Giới thiệu về giảm chiều dữ liệu 🚀
    **Giảm chiều dữ liệu** là quá trình chuyển đổi dữ liệu từ không gian có số chiều cao (với nhiều đặc trưng) sang không gian có số chiều thấp hơn mà vẫn giữ lại được những đặc trưng quan trọng nhất. Quá trình này giúp:
    - **Trực quan hóa dữ liệu 📊:** Hiển thị dữ liệu trong không gian 2D hoặc 3D, từ đó dễ dàng nhận biết cấu trúc, nhóm (clusters) hay các mối liên hệ giữa các mẫu.
    - **Giảm nhiễu và tăng tốc độ tính toán ⚡:** Loại bỏ những đặc trưng dư thừa, không cần thiết giúp mô hình học máy chạy nhanh hơn và tránh tình trạng quá khớp (overfitting).
    
    ### Các phương pháp giảm chiều dữ liệu phổ biến 🔍
    #### 1. PCA (Principal Component Analysis) 💡
    - **Nguyên lý:**  
      PCA tìm các thành phần chính (principal components) sao cho phần lớn phương sai của dữ liệu được giữ lại. Nó sử dụng biến đổi tuyến tính để chuyển đổi dữ liệu sang không gian mới với các thành phần độc lập.
    - **Ưu điểm:**  
      - Đơn giản, hiệu quả và dễ hiểu.
      - Giảm được số chiều mà vẫn giữ lại phần lớn thông tin quan trọng.
    - **Nhược điểm:**  
      - Là một phương pháp tuyến tính, không thể bắt được những quan hệ phi tuyến giữa các đặc trưng.
      - Đôi khi khó diễn giải ý nghĩa của các thành phần chính khi số chiều gốc quá lớn.
      
    #### 2. t-SNE (t-distributed Stochastic Neighbor Embedding) 🔥
    - **Nguyên lý:**  
      t-SNE trực quan hóa dữ liệu bằng cách chuyển đổi khoảng cách giữa các điểm trong không gian cao chiều thành xác suất, sau đó tái tạo lại các mối quan hệ này trong không gian 2D hoặc 3D. Phương pháp này giúp phát hiện các nhóm nhỏ (clusters) trong dữ liệu phi tuyến.
    - **Ưu điểm:**  
      - Rất hiệu quả trong việc trực quan hóa các tập dữ liệu phức tạp như hình ảnh, văn bản.
      - Giúp phát hiện các cấu trúc ẩn, nhóm (clusters) trong dữ liệu.
    - **Nhược điểm:**  
      - Tốc độ tính toán chậm khi xử lý số lượng mẫu lớn.
      - Kết quả có thể thay đổi mạnh tùy thuộc vào các tham số như perplexity và learning rate.
    
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
with tab2:
    st.title("Trực quan hóa PCA & t-SNE trên MNIST")
    # Nếu dữ liệu chưa được tải, hiển thị giao diện tải dữ liệu
    if not st.session_state.mnist_loaded:
        tai_du_lieu_MNIST()
    
    if st.session_state.mnist_loaded:
        X = st.session_state.X
        y = st.session_state.y
        st.write("Dữ liệu đã được tải thành công!")
    
        # Cho người dùng lựa chọn thuật toán
        option = st.radio(
            "Chọn thuật toán cần chạy:",
            ("PCA", "t-SNE"),
            help="Chọn PCA để thu gọn dữ liệu hoặc t-SNE để trực quan hóa không gian dữ liệu."
        )
        
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"Experiment_{option}_{timestamp}"
        mlflow.set_experiment(experiment_name)
        st.session_state.experiment_name = experiment_name
        st.write("Tên thí nghiệm:", experiment_name)
        
        if option == "PCA":
            st.subheader("Cấu hình PCA")
            n_components = st.slider(
                "Chọn số thành phần (n_components)",
                2, 100, 50,
                help="Số thành phần chính cần giữ lại sau khi thực hiện PCA."
            )
            
            if st.button("Chạy PCA", key="btn_pca"):
                X_pca = PCA(n_components=n_components).fit_transform(X_scaled)
                st.session_state.X_pca = X_pca
                st.success("PCA đã được tính!")
                st.subheader("Kết quả PCA")
                fig_pca = ve_bieu_do(X_pca[:, :2], y, "Trực quan hóa PCA")
                st.pyplot(fig_pca)
                
                # Logging với MLflow
                with mlflow.start_run():
                    mlflow.log_param("n_components", n_components)
                    fig_pca.savefig("pca_visualization.png")
                    mlflow.log_artifact("pca_visualization.png")
                st.success("Kết quả PCA đã được lưu với MLflow!")
        
        elif option == "t-SNE":
            st.subheader("Cấu hình t-SNE")
            perplexity = st.slider(
                "Chọn giá trị perplexity",
                5, 50, 30,
                help="Số lượng láng giềng được cân nhắc khi tính khoảng cách giữa các điểm."
            )
            learning_rate = st.slider(
                "Chọn learning_rate",
                10, 1000, 200,
                help="Tốc độ học khi tối ưu hóa không gian nhúng của t-SNE."
            )
            
            if st.button("Chạy t-SNE", key="btn_tsne"):
                # Nếu chưa có kết quả PCA, tự động tính PCA với n_components mặc định (50)
                if st.session_state.X_pca is None:
                    st.info("Chưa có kết quả PCA, tự động tính PCA với n_components = 50.")
                    X_pca = PCA(n_components=50).fit_transform(X_scaled)
                    st.session_state.X_pca = X_pca
                else:
                    X_pca = st.session_state.X_pca
                    
                tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
                X_tsne = tsne.fit_transform(X_pca)
                st.success("t-SNE đã được tính!")
                st.subheader("Kết quả t-SNE")
                fig_tsne = ve_bieu_do(X_tsne, y, "Trực quan hóa t-SNE")
                st.pyplot(fig_tsne)
                
                # Logging với MLflow
                with mlflow.start_run():
                    mlflow.log_param("perplexity", perplexity)
                    mlflow.log_param("learning_rate", learning_rate)
                    fig_tsne.savefig("tsne_visualization.png")
                    mlflow.log_artifact("tsne_visualization.png")
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
