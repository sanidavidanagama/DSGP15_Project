import os

# Folder structure without creating a root folder
FOLDERS = [
    # ML models
    "ml-models/mood/data",
    "ml-models/mood/notebooks",
    "ml-models/mood/model_export",

    "ml-models/image/data",
    "ml-models/image/notebooks",
    "ml-models/image/model_export",

    "ml-models/dev_classification/data",
    "ml-models/dev_classification/notebooks",
    "ml-models/dev_classification/model_export",

    "ml-models/emotion/data",
    "ml-models/emotion/notebooks",
    "ml-models/emotion/model_export",

    # Backend structure
    "backend/app/api/v1",
    "backend/app/core",
    "backend/app/services",
    "backend/app/models",
    "backend/app/schemas",
    "backend/app/utils",
    "backend/app/ml/mood_model",
    "backend/app/ml/image_model",
    "backend/app/ml/dev_model",
    "backend/app/ml/emotion_model",
    "backend/tests",

    "frontend"
]

def create_structure():
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)

        if folder.startswith("mobile"):
            continue

        gitkeep = os.path.join(folder, ".gitkeep")
        with open(gitkeep, "w") as f:
            pass

    print("Folder structure created successfully!")

if __name__ == "__main__":
    create_structure()
