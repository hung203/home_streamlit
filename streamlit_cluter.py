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

# Cache dữ liệu MNIST
@st.cache_data
def load_mnist_data(sample_size):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    total_samples = mnist.data.shape[0]
    if sample_size < total_samples:
        X, _, y, _ = train_test_split(X, y, train_size=sample_size / total_samples, random_state=42, stratify=y)
    return X, y, total_samples

# Tạo các tab
tab1, tab2, tab3 = st.tabs(["Lý thuyết về phân cụm", "Huấn luyện", "MLflow"])

# ------------------------
# Tab 1: Lý thuyết về phân cụm
# ------------------------
with tab1:    
    st.header("📌 Lý thuyết về phân cụm", divider="blue")
    st.subheader("1️⃣ Phân cụm là gì?")
    st.write("""
    Phân cụm (Clustering) là một kỹ thuật học máy không giám sát, nhằm nhóm các điểm dữ liệu có đặc điểm tương đồng vào cùng một cụm.  
    🔍 **Điểm khác biệt với phân loại:**  
    - Phân cụm không có nhãn trước (unsupervised).  
    - Tự động tìm ra cấu trúc ẩn trong dữ liệu dựa trên sự tương đồng.
    """)

    st.subheader("2️⃣ Các thuật toán phân cụm phổ biến", divider="blue")
    st.markdown("### 🔹 Thuật toán K-Means")
    st.write("K-Means là một trong những thuật toán phân cụm phổ biến nhất. Dưới đây là các bước thực hiện:")
    st.markdown("#### **Bước 1: Khởi tạo K tâm cụm ban đầu**")
    st.write("Chọn ngẫu nhiên **K điểm** từ tập dữ liệu làm tâm cụm ban đầu.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/K_Means_Example_Step_1.svg/1024px-K_Means_Example_Step_1.svg.png", caption="Minh họa bước 1")
    st.markdown("#### **Bước 2: Gán điểm dữ liệu vào cụm gần nhất**")
    st.write("- Tính **khoảng cách** (thường là khoảng cách Euclid) từ mỗi điểm dữ liệu đến từng tâm cụm.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/K_Means_Example_Step_2.svg/1024px-K_Means_Example_Step_2.svg.png", caption="Minh họa bước 2")
    st.markdown("#### **Bước 3: Cập nhật lại tâm cụm**")
    st.write("- Tính **trung bình tọa độ** của tất cả các điểm trong cùng một cụm.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/K_Means_Example_Step_3.svg/1024px-K_Means_Example_Step_3.svg.png", caption="Minh họa bước 3")
    st.markdown("#### **Bước 4: Lặp lại bước 2 và 3**")
    st.write("- Tiếp tục gán lại các điểm dữ liệu vào cụm gần nhất.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/K_Means_Example_Step_4.svg/1024px-K_Means_Example_Step_4.svg.png", caption="Minh họa bước 4")
    st.markdown("#### **Bước 5: Dừng thuật toán**")
    st.write("- Thuật toán dừng khi tâm cụm không còn thay đổi hoặc đạt số vòng lặp tối đa.")
    st.markdown("### 🎨 Lưu ý quan trọng về K-Means")
    st.write("""
    - **Ưu điểm:** Đơn giản, nhanh, hiệu quả với dữ liệu hình cầu.  
    - **Nhược điểm:** Cần chọn K trước, nhạy cảm với tâm ban đầu, không hiệu quả với cụm phức tạp.
    """)

    st.subheader("🔹 Thuật toán DBSCAN là gì?")
    st.write("""
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) là một thuật toán phân cụm dựa trên mật độ, phát hiện cụm bất kỳ hình dạng và nhiễu.  
    - **Eps (ε):** Khoảng cách tối đa để hai điểm được coi là lân cận.  
    - **MinPts:** Số điểm tối thiểu để hình thành cụm.
    """)

    st.subheader("2️⃣ Các bước hoạt động của thuật toán DBSCAN", divider="blue")
    st.markdown("#### **Bước 1: Xác định các tham số Eps và MinPts**")
    st.write("- Chọn **Eps (ε)** và **MinPts** để xác định điểm lõi.")
    st.markdown("#### **Bước 2: Phân loại các điểm dữ liệu**")
    st.write("- **Điểm lõi:** Có ít nhất MinPts điểm trong bán kính Eps.  \n- **Điểm biên:** Trong Eps của điểm lõi.  \n- **Điểm nhiễu:** Không thuộc cụm nào.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/DBSCAN-Illustration.svg/1280px-DBSCAN-Illustration.svg.png")
    st.markdown("#### **Bước 3: Xây dựng cụm từ các điểm lõi**")
    st.write("- Bắt đầu từ điểm lõi, mở rộng cụm với các điểm lân cận.")
    st.image("https://cdn.analyticsvidhya.com/wp-content/uploads/2020/03/db12.png")
    st.markdown("#### **Bước 4: Xử lý các điểm chưa được gán nhãn**")
    st.write("- Tiếp tục tạo cụm mới hoặc đánh dấu nhiễu.")
    st.markdown("#### **Bước 5: Dừng thuật toán**")
    st.write("- Dừng khi tất cả điểm được xử lý.")
    st.markdown("### 🎨 Lưu ý quan trọng về DBSCAN")
    st.write("""
    - **Ưu điểm:** Không cần chọn số cụm, phát hiện nhiễu.  
    - **Nhược điểm:** Nhạy cảm với Eps và MinPts, khó với dữ liệu chiều cao.
    """)

    st.subheader("3️⃣ Đánh giá chất lượng phân cụm")
    st.write("Các chỉ số đánh giá:")
    st.markdown("- **Silhouette Score**: Đo mức độ tách biệt giữa các cụm.")
    st.image("image/Screenshot 2025-03-03 084601.png")
    st.markdown("- **Adjusted Rand Index (ARI)**: So sánh với nhãn thực tế.")
    st.image("image/Screenshot 2025-03-03 084611.png")
    st.markdown("- **Davies-Bouldin Index**: Đánh giá sự tương đồng giữa các cụm.")
    st.image("image/Screenshot 2025-03-03 084626.png")

# ------------------------
# Tab 2: Huấn luyện
# ------------------------
with tab2:
    st.header("1. Chọn kích thước tập huấn luyện")

    if "mnist_loaded" not in st.session_state:
        st.session_state.mnist_loaded = False
        st.session_state.total_samples = 70000

    # Fragment cho tải dữ liệu
    @st.fragment
    def load_data_interface():
        sample_size = st.number_input(
            "Chọn số lượng mẫu dữ liệu sử dụng", 
            min_value=1000, 
            max_value=st.session_state.total_samples, 
            value=10000, 
            step=1000
        )
        if st.button("Tải dữ liệu MNIST"):
            X, y, total_samples = load_mnist_data(sample_size)
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.X_valid = X_valid
            st.session_state.y_valid = y_valid
            st.session_state.mnist_loaded = True
            st.session_state.selected_sample_size = sample_size
            st.write(f"Dữ liệu MNIST đã được tải với {sample_size} mẫu!")

    load_data_interface()

    # Hiển thị hình ảnh minh họa
    st.subheader("Ví dụ một vài hình ảnh minh họa")
    if st.session_state.mnist_loaded:
        X = st.session_state.X
        y = st.session_state.y

        # Fragment cho hiển thị ảnh
        @st.fragment
        def show_example_images():
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

        show_example_images()
    else:
        st.warning("Vui lòng tải dữ liệu trước khi hiển thị hình ảnh!")

    st.header("Huấn luyện và đánh giá mô hình")
    model_choice = st.selectbox("Chọn mô hình phân cụm", ["K-means", "DBSCAN"], key="model_choice_cluster")
    
    if model_choice == "K-means":
        n_clusters = st.number_input("Chọn số lượng clusters", min_value=2, max_value=20, value=10, step=1)
    elif model_choice == "DBSCAN":
        eps = st.number_input("Chọn giá trị eps", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
        min_samples = st.number_input("Chọn số mẫu tối thiểu", min_value=1, max_value=20, value=5, step=1)

    experiment_name = st.text_input(
        "Nhập tên cho thí nghiệm MLflow", 
        value=f"{model_choice}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    # Fragment cho huấn luyện mô hình
    @st.fragment
    def train_model():
        if st.button("Huấn luyện mô hình"):
            if not st.session_state.mnist_loaded:
                st.error("Vui lòng tải dữ liệu trước khi huấn luyện mô hình!")
            else:
                X_train_used = st.session_state.X_train
                y_train_used = st.session_state.y_train
                X_valid = st.session_state.X_valid
                y_valid = st.session_state.y_valid

                with st.spinner("Đang huấn luyện mô hình..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    mlflow.set_experiment(experiment_name)
                    with mlflow.start_run():
                        mlflow.log_param("experiment_name", experiment_name)
                        mlflow.log_param("model", model_choice)

                        if model_choice == "K-means":
                            mlflow.log_param("n_clusters", n_clusters)
                            status_text.text("Khởi tạo mô hình K-means...")
                            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=0)
                            progress_bar.progress(10)

                            status_text.text("Đang huấn luyện K-means trên tập huấn luyện...")
                            model.fit(X_train_used)
                            progress_bar.progress(50)

                            status_text.text("Dự đoán trên tập kiểm tra...")
                            y_pred = model.predict(X_valid)
                            progress_bar.progress(70)

                            status_text.text("Đang đánh giá hiệu suất mô hình...")
                            ari = adjusted_rand_score(y_valid, y_pred)
                            sil_score = silhouette_score(X_valid, y_pred) if len(np.unique(y_pred)) > 1 else -1
                            db_index = davies_bouldin_score(X_valid, y_pred) if len(np.unique(y_pred)) > 1 else -1
                            nmi = normalized_mutual_info_score(y_valid, y_pred)
                            progress_bar.progress(100)

                        elif model_choice == "DBSCAN":
                            mlflow.log_param("eps", eps)
                            mlflow.log_param("min_samples", min_samples)
                            status_text.text("Khởi tạo mô hình DBSCAN...")
                            model = DBSCAN(eps=eps, min_samples=min_samples)
                            progress_bar.progress(10)

                            status_text.text("Đang phân cụm dữ liệu với DBSCAN...")
                            model.fit(X_train_used)
                            progress_bar.progress(60)

                            status_text.text("Đang đánh giá hiệu suất mô hình...")
                            y_pred = model.labels_
                            ari = adjusted_rand_score(y_train_used, y_pred)
                            sil_score = silhouette_score(X_train_used, y_pred) if len(np.unique(y_pred)) > 1 else -1
                            db_index = davies_bouldin_score(X_train_used, y_pred) if len(np.unique(y_pred)) > 1 else -1
                            nmi = normalized_mutual_info_score(y_train_used, y_pred)
                            progress_bar.progress(100)

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

                st.success("Huấn luyện mô hình hoàn tất!")
                st.session_state.experiment_name = experiment_name

    train_model()

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
        X_vis = st.session_state.X_valid if model_choice == "K-means" else st.session_state.X_train
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_vis)

        df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Cụm": labels.astype(str)
        })
        df["Cụm"] = df["Cụm"].replace("-1", "Nhiễu")

        fig = px.scatter(
            df, x="PC1", y="PC2", color="Cụm",
            title=f"Trực quan phân cụm với {model_choice}",
            labels={"PC1": "Thành phần chính 1", "PC2": "Thành phần chính 2"},
            color_discrete_sequence=px.colors.qualitative.T10 if len(unique_labels) <= 10 else px.colors.qualitative.Dark24
        )
        fig.update_layout(
            legend_title_text="Cụm", title_font_size=14,
            xaxis_title_font_size=12, yaxis_title_font_size=12,
            legend=dict(x=1.05, y=1)
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Tab 3: MLflow
# ------------------------
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
