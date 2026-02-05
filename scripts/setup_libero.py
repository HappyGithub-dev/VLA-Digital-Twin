import os
from huggingface_hub import snapshot_download


def download_data():
    # Target directory for your data
    repo_id = "openvla/modified_libero_rlds"
    local_dir = os.path.join(os.getcwd(), "data")

    print(f"🚀 Downloading {repo_id} to {local_dir}...")

    # Download the dataset shards
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=["libero_spatial_no_noops/*"]
    )
    print("✅ Download complete.")


if __name__ == "__main__":
    download_data()