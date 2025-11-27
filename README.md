# Emotion Recognition and Creativity Assessment in Children's Drawings


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
This project aims to **analyze children's drawings** using **deep learning models** to detect emotions and assess creativity. The system is designed to provide insights into children's emotional states and creative expression through automated image analysis.

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
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Emotion Detection**: Recognize emotions from children's drawings using CNN-based models.
- **Creativity Assessment**: Evaluate creative aspects of drawings based on color usage, patterns, and shapes.
- **Interactive Dashboard**: View analysis results and visualizations.
- **Modular Structure**: Easily extendable to include more features or datasets.


---

## Dataset

This project uses the KIDO Children Drawing Dataset to train and evaluate the model.

Source: 
[Kaggle](https://www.kaggle.com/datasets/serdarciftci/kido-children-drawing-dataset) |
[Github](https://github.com/serdarciftci/KIDO) |
[IEEE Explorer](https://ieeexplore.ieee.org/abstract/document/11151239)


### Directory Structure

Once the project is cloned, all dataset files are located in: `ml-models/dataset`

### Format

Images: `.png`

Labels: `.csv` files mapping images to their emotion categories


---

## Project Structure


```
ml-models/                       # ML model development
├── mood/
│   ├── data/                     # Raw / processed data
│   ├── notebooks/                # Training notebooks / experiments
│   └── model_export/             # Trained models, tokenizers, etc.
├── image/
│   ├── data/
│   ├── notebooks/
│   └── model_export/
├── dev_classification/
│   ├── data/
│   ├── notebooks/
│   └── model_export/
└── emotion/
    ├── data/
    ├── notebooks/
    └── model_export/

backend/                          # FastAPI backend
├── app/
│   ├── main.py                   # Entry point for FastAPI
│   ├── api/v1/                   # API endpoints
│   ├── services/                 # Business logic
│   ├── models/                   # Database models 
│   ├── schemas/                  # Pydantic request/response models
│   ├── utils/                    # Helper functions
│   └── ml/                       # Inference only
│       ├── mood_model/
│       ├── image_model/
│       ├── dev_model/
│       └── emotion_model/
└── tests/                        # Backend tests

frontend/                           

```

---

## Installation

1. Install uv Library:
   ```bash
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   
   Set Path
   ```bash
   set Path=C:\Users\User\.local\bin;%Path%   # (cmd)
   $env:Path = "C:\Users\User\.local\bin;$env:Path"  # (powershell)
   ```
   
   Check version 
   ```bash
   uv --version
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/sanidavidanagama/DSGP15_Project.git
   cd DSGP15_Project
   ```

---

## Usage

1. Create a virtual environment:
   ```bash
   uv venv
   ```
   Activate virtual environment
   ```bash
   activate .venv\Scripts\activate
   ```

2. Initialize project
   ```bash
   uv sync
   ```

3. Select Branch

   *Steps*
   - Select remote from the top git branch
   - feature
   - select the respective branch
   - Checkout


4. Data Preparation:
    ```bash
   python ml-models/data_ingestion.py
   ```

---

## Model Details

This project uses four main machine learning components:

1. **Emotion Analysis**  
   Identifies the emotion expressed in a child's drawing based on colors, shapes, and visual patterns.

2. **Mood Detection**  
   Determines the overall mood of the artwork by analyzing tone, brightness, and composition.

3. **Image Feature Extraction**  
   Extracts important visual features from drawings, such as textures, edges, and color distribution, which are used by other components.

4. **Developmental Stage Assessment**  
   Evaluates drawings to estimate the child's artistic and cognitive development stage based on structure, object representation, and spatial layout.


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
