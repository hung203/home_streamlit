
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu
df = pd.read_csv("G:/ML/MLFlow/my_env/titanic.csv")

# Kiểm tra giá trị thiếu
missing_values_before = df.isnull().sum()

df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Cabin"].fillna("Unknown", inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Kiểm tra và xử lý dữ liệu trùng lặp
duplicates_before = df.duplicated().sum()
df.drop_duplicates(inplace=True)

# Chuyển đổi biến phân loại thành số 
label_enc = LabelEncoder()
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col] = label_enc.fit_transform(df[col])

# Chia dữ liệu train/valid/test
df_train, df_temp = train_test_split(df, test_size=0.3, random_state=42)
df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

# Logging với MLflow
mlflow.set_experiment("Titanic_Preprocessing")
with mlflow.start_run():
    mlflow.log_param("missing_values_before", missing_values_before.to_dict())
    mlflow.log_param("missing_values_after", df.isnull().sum().to_dict())
    mlflow.log_param("duplicates_before", duplicates_before)
    mlflow.log_param("duplicates_after", df.duplicated().sum())
    mlflow.log_param("train_size", len(df_train))
    mlflow.log_param("valid_size", len(df_valid))
    mlflow.log_param("test_size", len(df_test))
    mlflow.log_param("random_state", 42)

    # Lưu và log các tập dữ liệu
    df_train.to_csv("train.csv", index=False)
    df_valid.to_csv("valid.csv", index=False)
    df_test.to_csv("test.csv", index=False)

    mlflow.log_artifact("train.csv")
    mlflow.log_artifact("valid.csv")
    mlflow.log_artifact("test.csv")

print("Tiền xử lý hoàn tất và đã lưu các tập dữ liệu train, valid, test.")
# Lưu dữ liệu đã xử lý
# df.to_csv("titanic_cleaned.csv", index=False)
# print("Tiền xử lý hoàn tất và đã lưu vào titanic_cleaned.csv")
