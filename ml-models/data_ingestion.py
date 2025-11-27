import os
import shutil
import kagglehub

# ============================================================
# CONFIG
# ============================================================

# Folder where the dataset will be stored
DATASET_FOLDER = "dataset"

# Kaggle dataset ID
KAGGLE_DATASET = "serdarciftci/kido-children-drawing-dataset"


# ============================================================
# CREATE DATASET FOLDER IF NOT EXISTS
# ============================================================

def ensure_dataset_folder():
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    print(f"[OK] Ensured dataset folder: {DATASET_FOLDER}")


# ============================================================
# DOWNLOAD AND EXTRACT DATASET
# ============================================================

def download_and_extract_dataset():
    print("\n📥 Downloading dataset from Kaggle...")
    kaggle_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"✔ Dataset downloaded to: {kaggle_path}")

    # Extract dataset contents into the target folder
    print("\n📂 Copying dataset contents into project folder...")

    # If the kagglehub.download returns a zip file
    if os.path.isfile(kaggle_path):
        shutil.unpack_archive(kaggle_path, DATASET_FOLDER)
        print(f"✔ Dataset extracted to: {DATASET_FOLDER}")
        os.remove(kaggle_path)  # Delete original zip
        print(f"🗑 Deleted temporary file: {kaggle_path}")

    # If it returns a folder instead
    elif os.path.isdir(kaggle_path):
        for item in os.listdir(kaggle_path):
            s = os.path.join(kaggle_path, item)
            d = os.path.join(DATASET_FOLDER, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        shutil.rmtree(kaggle_path)  # Delete original folder
        print(f"🗑 Deleted temporary folder: {kaggle_path}")

    print("\n🎉 Dataset Imported Successfully!")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n========== DATA INGESTION STARTED ==========\n")
    ensure_dataset_folder()
    download_and_extract_dataset()
    print("\n========== DATA INGESTION COMPLETED ==========\n")
