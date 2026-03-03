# Mood 

Lightweight, production-ready repository for emotion / mood modeling from children's drawings. This branch contains data, experiments, and exported models for the mood classification component.

## Quick links
- Project root: [README.md](README.md)  
- Dataset helper: [`download_and_extract_dataset`](ml-models/data_ingestion.py) — see [ml-models/data_ingestion.py](ml-models/data_ingestion.py)  
- Mood assets:
  - Data folder: [ml-models/mood/data](ml-models/mood/data)  
  - Notebooks & experiments: [ml-models/mood/notebooks](ml-models/mood/notebooks)  
  - Exported models: [ml-models/mood/model_export](ml-models/mood/model_export)



## Models used
- Image models
  - ResNet-50 (primary model used for this project)
  - ResNet-18 (baseline / ablation)
- Text models
  - BERT (bert-base) — used for richer text/metadata experiments
  - DistilBERT (distilbert-base) — lightweight variant used across multiple experiments and quick iterations
- Multimodal / Fusion
  - Fusion model that combines image and text embeddings for final mood classification
  - End-to-end inference pipeline that accepts images and optional text/metadata, runs respective encoders, fuses features, and outputs mood probabilities

## Quick start
1. Create & activate virtualenv
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

2. Prepare dataset
```bash
python ml-models/data_ingestion.py
```

3. Run experiments / inference
- Open notebooks in [ml-models/mood/notebooks](ml-models/mood/notebooks) for training and evaluation.
- Use exported model files from [ml-models/mood/model_export](ml-models/mood/model_export) for inference.

## Usage 
- Training / evaluation: open the relevant notebook in [ml-models/mood/notebooks](ml-models/mood/notebooks).
- Inference (high level):
  - Load image encoder (ResNet-50) and text encoder (BERT / DistilBERT) as needed.
  - Pass inputs through encoders, run fusion head, produce probabilities.
  - Exported artifacts in ml-models/mood/model_export are ready for PyTorch/TensorFlow inference.

## Project structure
- ml-models/mood/data — datasets and splits  
- ml-models/mood/notebooks — experiments and visualizations  
- ml-models/mood/model_export — serialized models and artifacts

## Contributing
- Add experiments as notebooks in [ml-models/mood/notebooks](ml-models/mood/notebooks)
- Export final models to [ml-models/mood/model_export](ml-models/mood/model_export)
- Keep dataset ingestion reproducible via [ml-models/data_ingestion.py](ml-models/data_ingestion.py)

