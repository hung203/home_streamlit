import streamlit as st
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2

# Tiêu đề ứng dụng
st.title("Phân loại chữ số viết tay MNIST với Streamlit và MLflow")

tab1, tab2, tab3 = st.tabs([
    "Lý thuyết về phân cụm",
    "Huấn luyện",
    "Mlflow"
])

# ------------------------
# Bước 1: Xử lý dữ liệu
# ------------------------
with tab1:
    st.header("📌 Lý thuyết về phân cụm")
    
    st.subheader("1️⃣ Phân cụm là gì?")
    st.write("""
    Phân cụm (Clustering) là một kỹ thuật học máy không giám sát nhằm nhóm các điểm dữ liệu có đặc điểm tương đồng vào cùng một cụm. 
    Khác với phân loại, phân cụm không có nhãn trước mà tự động tìm ra cấu trúc trong dữ liệu.
    """)

    st.subheader("2️⃣ Các thuật toán phân cụm phổ biến")
    
    st.markdown("### 🔹 K-Means")
    st.write("""
    - K-Means là thuật toán phân cụm phổ biến, chia dữ liệu thành K nhóm dựa trên khoảng cách đến trọng tâm (centroid).
    - Quy trình:
        1. Chọn K cụm ban đầu.
        2. Gán mỗi điểm dữ liệu vào cụm gần nhất.""")
        
    st.image("image/Screenshot 2025-03-03 083928.png")     
    st.write("""3. Tính lại trọng tâm cho từng cụm.""")

    st.image("image/Screenshot 2025-03-03 084527.png")
    st.write("""4. Lặp lại cho đến khi các trọng tâm ổn định.
    """)
    
    st.markdown("### 🔹 DBSCAN")
    st.write("""
    - **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) là thuật toán phân cụm dựa trên mật độ.

    ##### Quy trình của thuật toán:
    1. Thuật toán lựa chọn một điểm dữ liệu bất kỳ. Sau đó tiến hành xác định các điểm lõi và điểm biên thông qua vùng lân cận epsilon bằng cách lan truyền theo liên kết chuỗi các điểm thuộc cùng một cụm.  
    2. Cụm hoàn toàn được xác định khi không thể mở rộng được thêm. Khi đó lặp lại đệ quy toàn bộ quá trình với điểm khởi tạo trong số các điểm dữ liệu còn lại để xác định một cụm mới.

    ### Ưu điểm:
    - Không cần xác định số cụm trước (không giống K-Means).  
    - Tốt trong việc phát hiện nhiễu.  

    ### Nhược điểm:
    - Nhạy cảm với tham số `eps` (Epsilon - Bán kính lân cận) và `min_samples` (Số lượng điểm tối thiểu).
    """)
    st.subheader("3️⃣ Đánh giá chất lượng phân cụm")
    st.write("Sau khi phân cụm, có nhiều cách đánh giá kết quả:")
    
    st.markdown("- **Silhouette Score**: Đo lường mức độ tách biệt giữa các cụm.")
    st.image("image/Screenshot 2025-03-03 084601.png")
    st.markdown("- **Adjusted Rand Index (ARI)**: So sánh phân cụm với nhãn thực tế (nếu có).")
    st.image("image/Screenshot 2025-03-03 084611.png")
    st.markdown("- **Davies-Bouldin Index**: Đánh giá sự tương đồng giữa các cụm.")
    st.image("image/Screenshot 2025-03-03 084626.png")

    
# ------------------------
# Bước 2: Huấn luyện và đánh giá mô hình (Phân cụm với K-means & DBSCAN)
# ------------------------
with tab2:
    st.header("1. Chọn kích thước tập huấn luyện")

    # Kiểm tra nếu dữ liệu đã được tải chưa
    if "mnist_loaded" not in st.session_state:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        st.session_state.total_samples = mnist.data.shape[0]  # Tổng số mẫu
        st.session_state.mnist_data = mnist  # Lưu dữ liệu gốc
        st.session_state.mnist_loaded = False  # Chưa tải mẫu cụ thể

    # Chọn số lượng mẫu sử dụng
    sample_size = st.number_input(
        "Chọn số lượng mẫu dữ liệu sử dụng", 
        min_value=1000, 
        max_value=st.session_state.total_samples, 
        value=st.session_state.total_samples, 
        step=1000
    )

    if st.button("Tải dữ liệu MNIST"):
        mnist = st.session_state.mnist_data
        X, y = mnist.data / 255.0, mnist.target.astype(int)
        
        # Chọn số lượng mẫu dữ liệu theo yêu cầu
        if sample_size < st.session_state.total_samples:
            X, _, y, _ = train_test_split(X, y, train_size=sample_size, random_state=42, stratify=y)
        
        # Lưu dữ liệu vào session_state
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.mnist_loaded = True
        st.session_state.selected_sample_size = sample_size
        st.write(f"Dữ liệu MNIST đã được tải với {sample_size} mẫu!")

    # Hiển thị hình ảnh minh họa
    st.subheader("Ví dụ một vài hình ảnh minh họa")
    
    if st.session_state.mnist_loaded:
        X = st.session_state.X
        y = st.session_state.y

        # Nút làm mới hình ảnh
        if st.button("🔄 Hiển thị ảnh mới"):
            st.session_state.example_images = random.sample(range(len(X)), 5)

        indices = st.session_state.get("example_images", random.sample(range(len(X)), 5))

        fig, axs = plt.subplots(1, 5, figsize=(12, 3))
        for i, idx in enumerate(indices):
            img = X[idx].reshape(28, 28)
            axs[i].imshow(img, cmap='gray')
            axs[i].axis('off')
            axs[i].set_title(f"Label: {y[idx]}")
        
        st.pyplot(fig)
    else:
        st.warning("Vui lòng tải dữ liệu trước khi hiển thị hình ảnh!")
        
    st.header("Huấn luyện và đánh giá mô hình")
    # Người dùng chọn mô hình phân cụm
    model_choice = st.selectbox("Chọn mô hình phân cụm", ["K-means", "DBSCAN"], key="model_choice_cluster")
    
    if model_choice == "K-means":
        n_clusters = st.number_input(
            "Chọn số lượng clusters", 
            min_value=2, 
            max_value=20, 
            value=10, 
            step=1,
            help="Số lượng clusters là số nhóm dữ liệu mà K-means sẽ tìm kiếm. Với MNIST, giá trị thông thường là 10."
        )
    elif model_choice == "DBSCAN":
        eps = st.number_input(
            "Chọn giá trị eps", 
            min_value=0.1, 
            max_value=10.0, 
            value=0.5, 
            step=0.1,
            help="Giá trị eps xác định khoảng cách tối đa giữa các điểm để được xem là cùng một cụm."
        )
        min_samples = st.number_input(
            "Chọn số mẫu tối thiểu", 
            min_value=1, 
            max_value=20, 
            value=5, 
            step=1,
            help="Số mẫu tối thiểu xung quanh một điểm cần có để điểm đó được xem là điểm lõi của một cụm."
        )
    
    # Nút huấn luyện
    if st.button("Huấn luyện mô hình"):    
        X_train_used = st.session_state.X_train
        y_train_used = st.session_state.y_train
        X_valid = st.session_state.X_valid
        y_valid = st.session_state.y_valid

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"Experiment_{model_choice}_{timestamp}"
        mlflow.set_experiment(experiment_name)
        # Lưu tên thí nghiệm vào session_state và hiển thị ra giao diện
        st.session_state.experiment_name = experiment_name
        st.write("Tên thí nghiệm:", experiment_name)
        with mlflow.start_run():
            mlflow.log_param("experiment_name", experiment_name)
            mlflow.log_param("model", model_choice)
            
            # Với K-means: huấn luyện trên tập train và dự đoán trên tập validation
            if model_choice == "K-means":
                mlflow.log_param("n_clusters", n_clusters)
                model = KMeans(n_clusters=n_clusters, random_state=42)
                model.fit(X_train_used)
                y_pred = model.predict(X_valid)
                ari = adjusted_rand_score(y_valid, y_pred)
                
                # Kiểm tra số lượng cluster trước khi tính các chỉ số đánh giá
                if len(np.unique(y_pred)) > 1:
                    sil_score = silhouette_score(X_valid, y_pred)
                    db_index = davies_bouldin_score(X_valid, y_pred)
                else:
                    sil_score = -1
                    db_index = -1
                
                nmi = normalized_mutual_info_score(y_valid, y_pred)
            
            # Với DBSCAN: huấn luyện trên tập train (vì không hỗ trợ predict trên dữ liệu mới)
            elif model_choice == "DBSCAN":
                mlflow.log_param("eps", eps)
                mlflow.log_param("min_samples", min_samples)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                model.fit(X_train_used)
                y_pred = model.labels_  # Nhãn phân cụm trên tập train
                ari = adjusted_rand_score(y_train_used, y_pred)
                
                if len(np.unique(y_pred)) > 1:
                    sil_score = silhouette_score(X_train_used, y_pred)
                    db_index = davies_bouldin_score(X_train_used, y_pred)
                else:
                    sil_score = -1
                    db_index = -1
                
                nmi = normalized_mutual_info_score(y_train_used, y_pred)
            
            # Lưu kết quả và mô hình vào session_state
            st.session_state.model = model
            st.session_state.trained_model_name = model_choice
            st.session_state.train_ari = ari
            st.session_state.train_sil = sil_score
            st.session_state.train_nmi = nmi
            st.session_state.train_db = db_index
            
            mlflow.log_metric("ARI", ari)
            mlflow.log_metric("Silhouette", sil_score)
            mlflow.log_metric("NMI", nmi)
            mlflow.log_metric("DaviesBouldin", db_index)
            mlflow.sklearn.log_model(model, "model")
    
        st.session_state.experiment_name = experiment_name
        st.write("Tên thí nghiệm:", experiment_name)
    
    # Hiển thị kết quả sau khi huấn luyện
    if "train_ari" in st.session_state:
        if model_choice == "K-means":
            st.write(f"🔹 **Adjusted Rand Index (Validation):** {st.session_state.train_ari:.4f}")
        elif model_choice == "DBSCAN":
            st.write(f"🔹 **Adjusted Rand Index (Train):** {st.session_state.train_ari:.4f}")
        st.write(f"🔹 **Silhouette Score:** {st.session_state.train_sil:.4f}")
        st.write(f"🔹 **Normalized Mutual Information:** {st.session_state.train_nmi:.4f}")
        st.write(f"🔹 **Davies-Bouldin Index:** {st.session_state.train_db:.4f}")
        
        # ------------------------
        # Trực quan hoá phân cụm với PCA
        # ------------------------
        st.subheader("Trực quan hoá phân cụm")
        from sklearn.decomposition import PCA
        
        # Chọn tập dữ liệu phù hợp để trực quan hoá
        if model_choice == "K-means":
            X_vis = X_valid
        else:
            X_vis = X_train_used
        
        # Giảm chiều dữ liệu xuống 2D để trực quan hoá
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_vis)
        
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', s=10)
        ax.set_title(f"Trực quan phân cụm với {model_choice}")
        plt.colorbar(scatter, ax=ax)
        
        st.pyplot(fig)
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

            # Chọn thí nghiệm dựa trên TÊN thay vì ID
            selected_exp_name = st.selectbox(
                "🔍 Chọn thí nghiệm để xem chi tiết",
                options=[exp.name for exp in experiments]
            )

            # Lấy ID tương ứng với tên được chọn
            selected_exp_id = next(exp.experiment_id for exp in experiments if exp.name == selected_exp_name)

            # Lấy danh sách runs trong thí nghiệm đã chọn
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

                # Hiển thị metrics
                st.write("##### Metrics")
                st.json(run.data.metrics)

                # Hiển thị params
                st.write("##### Params")
                st.json(run.data.params)

                # Hiển thị artifacts sử dụng client.list_artifacts
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
