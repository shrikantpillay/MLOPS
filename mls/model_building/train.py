import joblib
import mlflow
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

# Initialize API
api = HfApi()

# 1. CORRECT FILENAME FOR YOUR TOURISM MODEL
model_path = "best_tourism_model_v1.joblib"
joblib.dump(best_model, model_path)

# Log the model artifact in MLflow
mlflow.log_artifact(model_path, artifact_path="model")
print(f"Model saved as artifact at: {model_path}")

# Upload to Hugging Face
repo_id = "shrikantpillay/MLOPS"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repository '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Repository '{repo_id}' not found. Creating new repository...")
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repository '{repo_id}' created.")

# Upload the correctly named tourism model
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Tourism model uploaded successfully!")
