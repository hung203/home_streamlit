import datetime
from sklearn.decomposition import PCA
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
import plotly.express as px
import plotly.graph_objects as go

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
    st.header("📌 Lý thuyết về phân cụm", divider="blue")

    # Phần 1: Phân cụm là gì?
    st.subheader("1️⃣ Phân cụm là gì?")
    st.write("""
    Phân cụm (Clustering) là một kỹ thuật học máy không giám sát, nhằm nhóm các điểm dữ liệu có đặc điểm tương đồng vào cùng một cụm.  
    🔍 **Điểm khác biệt với phân loại:**  
    - Phân cụm không có nhãn trước (unsupervised).  
    - Tự động tìm ra cấu trúc ẩn trong dữ liệu dựa trên sự tương đồng.
    """)

    # Phần 2: Các thuật toán phân cụm phổ biến
    st.subheader("2️⃣ Các thuật toán phân cụm phổ biến", divider="blue")

    # Thuật toán K-Means
    st.markdown("### 🔹 Thuật toán K-Means")
    st.write("K-Means là một trong những thuật toán phân cụm phổ biến nhất. Dưới đây là các bước thực hiện:")

    # Bước 1
    st.markdown("#### **Bước 1: Khởi tạo K tâm cụm ban đầu**")
    st.write("Chọn ngẫu nhiên **K điểm** từ tập dữ liệu làm tâm cụm ban đầu.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/K_Means_Example_Step_1.svg/1024px-K_Means_Example_Step_1.svg.png",
            caption="Minh họa bước 1", use_container_width=True)

    # Bước 2
    st.markdown("#### **Bước 2: Gán điểm dữ liệu vào cụm gần nhất**")
    st.write("""
    - Tính **khoảng cách** (thường là khoảng cách Euclid) từ mỗi điểm dữ liệu đến từng tâm cụm.  
    - Gán mỗi điểm vào cụm có **tâm gần nhất**.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/K_Means_Example_Step_2.svg/1024px-K_Means_Example_Step_2.svg.png",
            caption="Minh họa bước 2", use_container_width=True)

    # Bước 3
    st.markdown("#### **Bước 3: Cập nhật lại tâm cụm**")
    st.write("""
    - Tính **trung bình tọa độ** của tất cả các điểm trong cùng một cụm.  
    - Đặt giá trị trung bình này làm **tâm cụm mới**.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/K_Means_Example_Step_3.svg/1024px-K_Means_Example_Step_3.svg.png",
            caption="Minh họa bước 3", use_container_width=True)

    # Bước 4
    st.markdown("#### **Bước 4: Lặp lại bước 2 và 3**")
    st.write("""
    - Tiếp tục gán lại các điểm dữ liệu vào cụm gần nhất dựa trên tâm cụm mới.  
    - Cập nhật lại tâm cụm sau mỗi lần gán. 
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/K_Means_Example_Step_4.svg/1024px-K_Means_Example_Step_4.svg.png",
            caption="Minh họa bước 4", use_container_width=True)

    # Bước 5
    st.markdown("#### **Bước 5: Dừng thuật toán**")
    st.write("""
    Thuật toán dừng khi:  
    - Các tâm cụm không còn thay đổi, hoặc  
    - Đạt số vòng lặp tối đa đã thiết lập.
    """)

    # Lưu ý quan trọng về K-Means
    st.markdown("### 🎨 Lưu ý quan trọng về K-Means")
    st.write("""
    - **Ưu điểm:** Đơn giản, nhanh, hiệu quả với dữ liệu hình cầu.  
    - **Nhược điểm:**  
        - Cần chọn số cụm K trước.  
        - Nhạy cảm với giá trị ban đầu của tâm cụm.  
        - Không hiệu quả với cụm có hình dạng phức tạp.
    """)


    st.subheader("🔹 Thuật toán DBSCAN là gì?")
    st.write("""
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) là một thuật toán phân cụm dựa trên mật độ, có khả năng phát hiện các cụm có hình dạng bất kỳ và xác định nhiễu (noise) trong dữ liệu.  
    Khác với K-Means, DBSCAN không yêu cầu xác định số cụm trước và hoạt động dựa trên hai tham số chính:  
    - **Eps (ε):** Khoảng cách tối đa để hai điểm được coi là "lân cận".  
    - **MinPts:** Số điểm tối thiểu cần thiết để hình thành một cụm.
    """)

    # Các bước hoạt động của DBSCAN
    st.subheader("2️⃣ Các bước hoạt động của thuật toán DBSCAN", divider="blue")

    # Bước 1
    st.markdown("#### **Bước 1: Xác định các tham số Eps và MinPts**")
    st.write("""
    - Chọn giá trị **Eps (ε):** Khoảng cách tối đa giữa hai điểm để chúng được coi là thuộc cùng một vùng mật độ.  
    - Chọn giá trị **MinPts:** Số điểm tối thiểu cần thiết trong bán kính Eps để một điểm được coi là **điểm lõi (core point)**.
    """)

    # Bước 2
    st.markdown("#### **Bước 2: Phân loại các điểm dữ liệu**")
    st.write("""
    Dựa trên Eps và MinPts, các điểm dữ liệu được phân loại thành ba loại:  
    - **Điểm lõi (Core Point):**  
        - Một điểm có **ít nhất MinPts điểm** (bao gồm chính nó) trong bán kính Eps.  
        - Đây là điểm trung tâm của một cụm.  
    - **Điểm biên (Border Point):**  
        - Một điểm không phải là điểm lõi, nhưng nằm trong bán kính Eps của ít nhất một điểm lõi.  
        - Điểm biên thuộc về cụm nhưng không mở rộng cụm.  
    - **Điểm nhiễu (Noise Point):**  
        - Một điểm không phải là điểm lõi và không nằm trong bán kính Eps của bất kỳ điểm lõi nào.  
        - Điểm nhiễu không thuộc cụm nào.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/DBSCAN-Illustration.svg/1280px-DBSCAN-Illustration.svg.png")
    
    # Bước 3
    st.markdown("#### **Bước 3: Xây dựng cụm từ các điểm lõi**")
    st.write("""
    - Bắt đầu từ một điểm lõi chưa được gán nhãn.  
    - Tạo một cụm mới và thêm điểm lõi này vào cụm.  
    - Tìm tất cả các điểm lân cận (trong bán kính Eps) của điểm lõi này:  
        - Nếu một điểm lân cận cũng là điểm lõi, tiếp tục mở rộng cụm bằng cách thêm các điểm lân cận của điểm lõi mới.  
        - Nếu một điểm lân cận là điểm biên, thêm nó vào cụm nhưng không mở rộng thêm từ điểm biên.  
    - Lặp lại quá trình này cho đến khi không còn điểm lân cận nào có thể thêm vào cụm.
    """)
    st.image("https://cdn.analyticsvidhya.com/wp-content/uploads/2020/03/db12.png")

    # Bước 4
    st.markdown("#### **Bước 4: Xử lý các điểm chưa được gán nhãn**")
    st.write("""
    - Chọn một điểm chưa được gán nhãn khác:  
        - Nếu là điểm lõi, tạo một cụm mới và lặp lại bước 3.  
        - Nếu là điểm biên, nó sẽ được gán vào cụm gần nhất (nếu có).  
        - Nếu là điểm nhiễu, đánh dấu điểm này là nhiễu và không gán vào cụm nào.
    """)

    # Bước 5
    st.markdown("#### **Bước 5: Dừng thuật toán**")
    st.write("""
    - Thuật toán dừng khi tất cả các điểm dữ liệu đã được xử lý:  
        - Mỗi điểm được gán vào một cụm (nếu là điểm lõi hoặc điểm biên), hoặc  
        - Được đánh dấu là nhiễu (nếu là điểm nhiễu).
    """)

    # Lưu ý quan trọng về DBSCAN
    st.markdown("### 🎨 Lưu ý quan trọng về DBSCAN")
    st.write("""
    - **Ưu điểm:**  
        - Không cần xác định số cụm trước.  
        - Có thể phát hiện cụm có hình dạng bất kỳ (không giới hạn hình cầu như K-Means).  
        - Xác định được nhiễu (noise) trong dữ liệu.  
    - **Nhược điểm:**  
        - Nhạy cảm với việc chọn tham số Eps và MinPts.  
        - Không hiệu quả với dữ liệu có mật độ cụm không đồng đều.  
        - Có thể gặp khó khăn với dữ liệu chiều cao (curse of dimensionality).
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
        
        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra (validation)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
        
        # Lưu dữ liệu vào session_state
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_valid = X_valid
        st.session_state.y_valid = y_valid
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
    # Nhập tên cho thí nghiệm MLflow
    experiment_name = st.text_input(
        "Nhập tên cho thí nghiệm MLflow", 
        value="",
        help="Tên để lưu thí nghiệm trong MLflow. Nếu để trống, hệ thống sẽ tự tạo tên dựa trên thời gian."
    )
    if not experiment_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"{model_choice}_{timestamp}"
    
    # Nút huấn luyện
    # Trong phần "Nút huấn luyện" của tab2
    if st.button("Huấn luyện mô hình"):
        # Kiểm tra xem dữ liệu đã được tải chưa
        if not st.session_state.mnist_loaded:
            st.error("Vui lòng tải dữ liệu trước khi huấn luyện mô hình!")
        else:
            X_train_used = st.session_state.X_train
            y_train_used = st.session_state.y_train
            X_valid = st.session_state.X_valid
            y_valid = st.session_state.y_valid

            # Sử dụng st.spinner để hiển thị trạng thái huấn luyện
            with st.spinner("Đang huấn luyện mô hình..."):
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
                        
                        if len(np.unique(y_pred)) > 1:
                            sil_score = silhouette_score(X_valid, y_pred)
                            db_index = davies_bouldin_score(X_valid, y_pred)
                        else:
                            sil_score = -1
                            db_index = -1
                        
                        nmi = normalized_mutual_info_score(y_valid, y_pred)
                    
                    # Với DBSCAN: huấn luyện trên tập train
                    elif model_choice == "DBSCAN":
                        mlflow.log_param("eps", eps)
                        mlflow.log_param("min_samples", min_samples)
                        model = DBSCAN(eps=eps, min_samples=min_samples)
                        model.fit(X_train_used)
                        y_pred = model.labels_
                        ari = adjusted_rand_score(y_train_used, y_pred)
                        
                        if len(np.unique(y_pred)) > 1:
                            sil_score = silhouette_score(X_train_used, y_pred)
                            db_index = davies_bouldin_score(X_valid, y_pred)
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
            
            # Thông báo huấn luyện hoàn tất
            st.success("Huấn luyện mô hình hoàn tất!")
        
            st.session_state.experiment_name = experiment_name
    
    # Hiển thị kết quả sau khi huấn luyện
    if "train_ari" in st.session_state:
        st.write("### Kết quả phân cụm")
        labels = st.session_state.model.labels_ if model_choice == "DBSCAN" else st.session_state.model.predict(st.session_state.X_valid)
        unique_labels = np.unique(labels)
        st.write(f"**Số lượng cụm tìm thấy:** {len(unique_labels) if -1 not in unique_labels else len(unique_labels) - 1}")
        cluster_counts = pd.Series(labels).value_counts()
        cluster_df = pd.DataFrame({"Cụm": cluster_counts.index, "Số lượng điểm": cluster_counts.values})
        st.dataframe(cluster_df)
        if -1 in labels:
            noise_ratio = (labels == -1).mean() * 100
            st.write(f"**Tỷ lệ nhiễu:** {noise_ratio:.2f}%")
        if model_choice == "K-means":
            st.write(f"🔹 **Adjusted Rand Index (Validation):** {st.session_state.train_ari:.4f}")
        elif model_choice == "DBSCAN":
            st.write(f"🔹 **Adjusted Rand Index (Train):** {st.session_state.train_ari:.4f}")
        st.write(f"🔹 **Silhouette Score:** {st.session_state.train_sil:.4f}")
        st.write(f"🔹 **Normalized Mutual Information:** {st.session_state.train_nmi:.4f}")
        st.write(f"🔹 **Davies-Bouldin Index:** {st.session_state.train_db:.4f}")
        
        # Trực quan hoá phân cụm với PCA
        st.subheader("Trực quan hoá phân cụm")

        # Chọn tập dữ liệu phù hợp để trực quan hoá
        if model_choice == "K-means":
            X_vis = st.session_state.X_valid
        else:
            X_vis = st.session_state.X_train

        # Giảm chiều dữ liệu xuống 2D để trực quan hoá
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_vis)

        # Lấy nhãn phân cụm
        labels = st.session_state.model.labels_ if model_choice == "DBSCAN" else st.session_state.model.predict(st.session_state.X_valid)
        unique_labels = np.unique(labels)

        # Tạo DataFrame để sử dụng với plotly
        df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Cụm": labels.astype(str)  # Chuyển nhãn thành chuỗi để dễ hiển thị
        })

        # Đổi nhãn nhiễu thành "Nhiễu" nếu có
        df["Cụm"] = df["Cụm"].replace("-1", "Nhiễu")

        # Vẽ biểu đồ phân tán tương tác
        fig = px.scatter(
            df,
            x="PC1",
            y="PC2",
            color="Cụm",
            title=f"Trực quan phân cụm với {model_choice}",
            labels={"PC1": "Thành phần chính 1", "PC2": "Thành phần chính 2"},
            color_discrete_sequence=px.colors.qualitative.T10 if len(unique_labels) <= 10 else px.colors.qualitative.Dark24
        )

        # Cập nhật layout
        fig.update_layout(
            legend_title_text="Cụm",
            title_font_size=14,
            xaxis_title_font_size=12,
            yaxis_title_font_size=12,
            legend=dict(x=1.05, y=1)
        )

        # Hiển thị biểu đồ
        st.plotly_chart(fig, use_container_width=True)
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
