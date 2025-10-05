# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
from tourism_project.config import HF_REPO_ID


repo_id = HF_REPO_ID
repo_type = "dataset"

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "/content/tourism_project/data/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=["CustomerID"], inplace=True)

# Encoding the categorical  columns
label_encoder = LabelEncoder()
df['Occupation'] = label_encoder.fit_transform(df['Occupation'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['TypeofContact'] = label_encoder.fit_transform(df['TypeofContact'])
df['MaritalStatus'] = label_encoder.fit_transform(df['MaritalStatus'])
df['Designation'] = label_encoder.fit_transform(df['Designation']) 
df['ProductPitched'] = label_encoder.fit_transform(df['ProductPitched'])
df['CityTier'] = label_encoder.fit_transform(df['CityTier'])
df['OwnCar'] = label_encoder.fit_transform(df['OwnCar'])
df['Passport'] = label_encoder.fit_transform(df['Passport'])

target_col = 'ProdTaken'

df.drop_duplicates(inplace=True)
df.dropna(subset=["ProdTaken"], inplace=True)  # Ensure target column is present
df.fillna(df.median(numeric_only=True), inplace=True)
print(f"Data cleaned. Remaining columns: {list(df.columns)}")

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id = HF_REPO_ID,
        repo_type=repo_type,
    )
