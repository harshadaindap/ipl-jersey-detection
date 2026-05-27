# IPL Jersey Detection

Classical ML pipeline for IPL team identification from cricket match images — built with hand-crafted features (HOG, LBP, color histograms), no CNNs.

---

## Overview

This project identifies the presence of cricket players in IPL match images and classifies each detected player into one of the 10 IPL franchises based on visual cues like jersey color, patterns, and logos. The image is divided into an **8×8 grid (64 cells)** and each cell is independently classified.

The key constraint: **no convolutional neural networks or any deep-learning feature extractors are used.** All features are hand-crafted from classical computer vision techniques, and classification is done using traditional ML models.

This was built as a course project for **Programming for Machine Learning and Data Science (PML)** at CMInDS, IIT Bombay.

---

## Problem Statement

Given an 800×600 image from an IPL match:

1. Divide the image into an 8×8 grid (64 cells, each 100×75 pixels)
2. For each cell, predict one of 11 labels:

| Label | Team |
|-------|------|
| 0 | No team / background |
| 1 | Chennai Super Kings (CSK) |
| 2 | Delhi Capitals (DC) |
| 3 | Gujarat Titans (GT) |
| 4 | Kolkata Knight Riders (KKR) |
| 5 | Lucknow Super Giants (LSG) |
| 6 | Mumbai Indians (MI) |
| 7 | Punjab Kings (PBKS) |
| 8 | Rajasthan Royals (RR) |
| 9 | Royal Challengers Bengaluru (RCB) |
| 10 | Sunrisers Hyderabad (SRH) |

If a cell contains multiple players from different teams, predicting any one of them is acceptable.

---

## Approach

### 1. Dataset Creation
- Collected cricket images from public sources (IPL website, ESPNcricinfo, Cricbuzz, Wikipedia Commons)
- Preprocessed all images to a uniform **800×600 resolution at 4:3 aspect ratio**
- Annotated using bounding boxes, then mapped to 8×8 grid labels
- Ensured each franchise has at least 100 cell-level instances
- Included background-only images (empty pitch, crowd) for class 0

### 2. Feature Engineering
Per cell (100×75 pixels), the following hand-crafted features are extracted:

- **Color features**: HSV color histograms, dominant color via k-means, color moments (mean, std, skewness)
- **Texture features**: HOG (Histogram of Oriented Gradients), LBP (Local Binary Patterns)
- **Team-color priors**: Distance of dominant cell color to each team's reference jersey color
- **Spatial features**: Cell row/column position, edge density (Canny)

Final feature vector per cell is standardized with `StandardScaler`.

### 3. Modeling
Two-stage classification pipeline:

- **Stage 1** — Binary classifier (player vs. no player) to handle class imbalance
- **Stage 2** — 10-class team classifier on cells flagged as containing a player

Models evaluated: Logistic Regression, Random Forest, XGBoost, SVM. Final model selected based on macro-F1 score on the validation set.

### 4. Evaluation
Performance is reported on training, test, and (where applicable) hold-out sets using:
- Overall and per-class accuracy
- Macro-F1, weighted-F1
- Confusion matrix
- Visual overlay of predictions on sample images

---

## Repository Structure

```
ipl-jersey-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes              # Git LFS config
├── data/
│   ├── README.txt              # Image sources and licensing notes
│   ├── images/
│   │   ├── train/
│   │   └── test/
│   └── annotations.csv         # Per-image grid labels
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── preprocess.py           # Image resizing, aspect ratio correction
│   ├── annotate.py             # Bounding box to grid label conversion
│   ├── extract_features.py     # All feature extraction logic
│   ├── train.py                # Model training and selection
│   ├── predict.py              # Inference on new images
│   └── pipeline.py             # End-to-end pipeline (loads .pkl, runs prediction)
├── models/
│   └── model_<teamname>.pkl    # Final trained model
├── outputs/
│   └── predictions.csv         # Required output format
├── docs/
│   ├── presentation.pdf
│   └── video_link.txt
└── tests/
    └── test_pipeline.py
```

---

## Setup and Installation

### Prerequisites
- Python 3.9 or higher
- Git and Git LFS (for the image dataset)

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/ipl-jersey-detection.git
cd ipl-jersey-detection

# Pull LFS files (images)
git lfs pull

# Create a virtual environment
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Main libraries used:
- `numpy`, `pandas` — numerical and data handling
- `opencv-python` — image I/O and preprocessing
- `scikit-image` — HOG, LBP feature extraction
- `scikit-learn` — classical ML models and metrics
- `xgboost` — gradient boosting classifier
- `imbalanced-learn` — class imbalance handling
- `matplotlib`, `seaborn` — visualization
- `joblib` — model serialization

---

## How to Run

### 1. Preprocess raw images
```bash
python src/preprocess.py --input data/raw --output data/images
```

### 2. Extract features from labeled images
```bash
python src/extract_features.py --images data/images --annotations data/annotations.csv --output data/features.npz
```

### 3. Train the model
```bash
python src/train.py --features data/features.npz --output models/model_<teamname>.pkl
```

### 4. Run the pipeline on a test image
```bash
python src/pipeline.py --model models/model_<teamname>.pkl --image path/to/test_image.jpg
```

### 5. Generate the final predictions CSV
```bash
python src/predict.py --model models/model_<teamname>.pkl --data data/images --output outputs/predictions.csv
```

---

## Output Format

The final `predictions.csv` follows this schema:

| ImageFileName | TrainOrTest | c01 | c02 | ... | c64 |
|---|---|---|---|---|---|
| img_0001.jpg | Train | 0 | 0 | ... | 1 |
| img_0002.jpg | Test | 2 | 2 | ... | 0 |

Each cell value is an integer from 0 to 10 corresponding to the team label.

---

## Results

_To be filled in after final evaluation._

| Metric | Train | Test | Hold-out |
|---|---|---|---|
| Accuracy | — | — | — |
| Macro-F1 | — | — | — |
| Weighted-F1 | — | — | — |

Detailed per-class metrics and confusion matrices are available in `docs/results.md`.

---

## Key Challenges and Learnings

- **Color ambiguity**: Several teams share similar jersey colors (DC blue vs. MI blue, RR pink vs. RCB red accents) — required adding team-color prior features
- **Class imbalance**: Background cells (class 0) dominate the dataset — addressed with two-stage classification and class weighting
- **Lighting variation**: Day vs. night matches required color constancy normalization
- **Annotation overhead**: Grid-level labeling for 2000+ images was the largest time sink — built a custom annotation tool to speed this up
- **Feature selection**: Initial naive RGB histograms plateaued at ~40% accuracy; switching to HSV + HOG + LBP gave the largest jump

---

## Team

| Name | Roll Number |
|---|---|
| _Member 1_ | _Roll No._ |
| _Member 2_ | _Roll No._ |
| _Member 3_ | _Roll No._ |
| _Member 4_ | _Roll No._ |

---

## Acknowledgements

- **Course**: Programming for Machine Learning and Data Science (PML)
- **Institution**: CMInDS, Indian Institute of Technology Bombay
- Image sources: IPL official website, ESPNcricinfo, Cricbuzz, Wikipedia Commons (see `data/README.txt` for full attribution)

---

## License

This project is released for academic and educational purposes. Images in the dataset are used under fair-use provisions for non-commercial research; please refer to original sources for redistribution rights.
