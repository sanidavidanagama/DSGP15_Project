# INKIND - Data Science Group Project

*Interpreting Non-verbal Knowledge IN Drawings*

---

## Interpretable Analysis of Psychological Indicators in Children’s Drawings Using Deep Learning

**Second Year Group Project – BSc (Hons) Artificial Intelligence and Data Science**  
**In collaboration with:** Robert Gordon University, Aberdeen, Scotland, UK  
**In partnership with:** Informatics Institute of Technology, Sri Lanka  
**Supervised by:** Mr. Prashan Rathnayaka

---

## Acknowledgements
This project is conducted in partnership with the Informatics Institute of Technology (Sri Lanka)  
and Robert Gordon University (Aberdeen, Scotland, UK).  
We thank our supervisor Mr. Prashan Rathnayaka for his guidance and support.

---

## Overview
This project provides an end-to-end system to analyze children's drawings using a set of **machine learning models** and a full **web application stack**:

- **ML model development** and experimentation.
- A **FastAPI backend** for inference, data processing, and APIs.
- A **Streamlit frontend** for interactive dashboards and user workflows.

The goal is to detect emotions, understand mood, and present insights to educators and parents.

---

## Team Members

| Student Name        | IIT ID   |
|---------------------|----------|
| Sanida Vidanagama   | 20231382 |
| Lidiya Rajapaksha   | 20240892 |
| Sanuli Dhanuge      | 20231350 |
| Kaviyan Ratneswaran | 20233020 |

---

## Table of Contents
- [Overview](#overview)
- [Team Members](#team-members)
- [System Components](#system-components)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Model Assets](#model-assets)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Contributing](#contributing)
- [License](#license)

---

## System Components

All components directly map to folders in this repository:

- **ML Models (development)** – `ml-models/`  
  Notebooks, data preparation, and training code for multiple models (mood, image, developmental classification, RAG, recommendation, etc.).

- **Backend API & Inference** – `backend/`  
  FastAPI backend, database layer, business logic, and deployed inference pipelines (using the trained models).

- **Frontend Application** – `frontend/`  
  Streamlit-based UI for teachers and researchers to upload drawings, run analyses, and explore dashboards.

---

## Project Structure

High-level view of the main folders:

```bash
ml-models/                         # ML model development
├── dev_classification/            # Developmental stage model (depreciated)
├── dia/                           # Drawing Indicator Analysis RAG & analysis
├── image/                         # Image preprocessing model
├── mood/                          # Mood / emotion model training
└── recommendation/                # Recommendation model

backend/                           # Application backend (FastAPI)
├── main.py
└── app/
    ├── core/                      # App config, settings
    ├── database/                  # DB models and CRUD logic
    ├── ml/                        # Inference-only model pipelines
    │   ├── dia_model/
    │   ├── image_model/
    │   ├── mood_model/
    │   └── recommendation_model/
    ├── models/                    # ORM models
    ├── routers/                   # API routes
    ├── schemas/                   # Pydantic request/response schemas
    ├── services/                  # Business logic services
    └── utils/                     # Helper utilities

frontend/                          # Streamlit frontend
├── app.py                         # Frontend entry point
├── pages/                         # Multi-page views
├── components/
├── services/                      # API client wrappers
└── utils/

uploads/
├── processed/                     # Processed image outputs
└── raw/                           # Raw uploaded drawings
```

---

## Installation & Setup

1. **Install `uv` (Python package/dependency manager)**  
   Follow the official [installation guide](https://docs.astral.sh/uv/getting-started/installation/) for your OS:  

2. **Clone the repository**

   ```bash
   git clone https://github.com/sanidavidanagama/DSGP15_Project.git
   cd DSGP15_Project
   ```

3. **Install dependencies with `uv`**

   From the project root:

   ```bash
   uv sync
   ```

   This will create and manage the virtual environment and install all dependencies defined in `pyproject.toml`.

4. **Configure environment**

   - Create a `.env` file as shown in [Configuration](#configuration).
   - Download and place the mood model data as described in [Model Assets](#model-assets).

---

## Configuration

The backend is configured via environment variables. Create a `.env` file in the project root (or configure these variables in your environment) with at least the following values:

```env
DATABASE_URL=sqlite:///./database.db
API_PREFIX=/api
DEBUG=True
ALLOWED_ORIGINS=https://localhost:3000, https://localhost:5173
GOOGLE_API_KEY=your_api_key
GEMINI_MODEL=gemini-3-flash-preview
ST_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_TOP_K=6
TF_ENABLE_ONEDNN_OPTS=0
PROCESSED_IMAGE_DIR=uploads/processed/
RAW_IMAGE_DIR=uploads/raw/
EMOTION_BERT_MODEL_DIR=path_to_image_dir
EMOTION_RESNET_MODEL_PATH=path_to_resnet_model
EMOTION_FUSION_MODEL_PATH=path_to_fusion_model
```

- Obtain your `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/app/api-keys).
- After downloading the model weights, replace `EMOTION_BERT_MODEL_DIR`, `EMOTION_RESNET_MODEL_PATH`, and `EMOTION_FUSION_MODEL_PATH` with **absolute paths** to those files/directories on your machine.
- `PROCESSED_IMAGE_DIR` and `RAW_IMAGE_DIR` should match the folders under `uploads/`.

---

## Model Assets

Some models require additional data and weights that are not stored directly in the repository.

### Mood / Emotion Model Data

Download the **`data`** folder for the mood model and place it under:

```text
backend/app/ml/mood_model/
```

Download from Google Drive:

- [data/](https://drive.google.com/drive/folders/1j0kaPqBl3NkMnQ9IBd0Oi34GHtZr-oHX?usp=drive_link)

After download, you should have (for example):

```bash
backend/app/ml/mood_model/
└── data/
    ├── fusion_model.pth
    ├── resnet50_emotion_model_cpu_optimized.pth
    └── saved_emotion_bert/
        ├── config.json
        ├── model.safetensors
        ├── vocab.txt
        └── ...
```

Make sure any other model paths (e.g., `EMOTION_BERT_MODEL_DIR`) in your `.env` file are consistent with where these files are stored.


---

## Running the Application

All commands below assume you are in the project root (`DSGP15_Project/`) and `uv sync` has already been run.

### Start the Backend (FastAPI)

```bash
cd backend
uv run main.py
```

This starts the FastAPI backend using the settings and models configured above.

### Start the Frontend (Streamlit)

Open a new terminal (still in the project root) and run:

```bash
cd frontend
uv run streamlit run app.py
```

Then open the URL shown in the terminal (by default, a `localhost` port) to access the UI.

Make sure the backend is running before you start using the frontend.

---

## Contributing

This project is maintained only by our four group members. When making updates:

- Work on a separate branch for your changes.
- Keep commits clear and meaningful.
- Make sure your code is clean, organized, and tested.
- Push your branch and let the team know before merging.

---

## License

This project is released under the [**MIT License**](LICENSE).  
Using this license ensures that our work remains protected while still allowing others to learn from or build upon it, as long as proper credit is given.
