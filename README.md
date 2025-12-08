# CSCI-635-01-Group-2
**An exploration of classifying art styles of famous works using various ML models.** <br /> <br />
Created for CSCI-635 (Introduction to Machine Learning) at RIT fall 2025, this project explores various supervised and unsupevised models for classification of art styles when given pictoral data of a painting.


## Table of Contents
- [Project Overiew](#CSCI-635-01-Group-2)
- [Table of Contents](#table-of-contents)
- [Abstract](#abstract)
- [Dev Setup](#dev-setup)
  - [System Requirements](#system-requirements)
  - [Cloning the Repository](#cloning-the-repository)
  - [Installing Dependencies](#installing-dependencies)
  - [Making a Local Development Branch](#making-a-local-development-branch)
- [Dataset Overview](#dataset-overview)
  - [Data Preparation Workflow](#data-preparation-workflow)
  - [Final Dataset Splits](#final-dataset-splits)
- [Project Structure Overview](#project-structure-overview)
- [Running the Models](#running-the-models)
  - [Running Classical Models](#running-classical-models)
  - [Running the CNN Model](#running-the-cnn-model)

---
# Abstract
This project investigates the problem of art style classification using a combination of classical machine-learning models and modern convolutional neural networks. Using five major styles from the WikiArt dataset (Abstract Expressionism, Baroque, Cubism, Impressionism, and Pop Art), how well different approaches can distinguish between visually complex and often overlapping artistic movements is explored. After preprocessing, balancing, and feature extraction, decision tree forests, k-nearest neighbors, and linear SVMs using 2048-dimensional ResNet50 embeddings are evaluated. These results are then compared to a lightweight convolutional neural network built on a frozen EfficientNetB0 backbone, with a test accuracy of >84%.

These experiments show that classical models tend to overfit highly expressive visual embeddings, whereas the CNN generalizes substantially better due to learned hierarchical features and in-model data augmentation. Among all models tested, the CNN achieved the strongest performance, with the highest validation and test MCC/accuracy scores, while the decision tree forest performed the weakest. Confusion matrices reveal that visually similar styles, particularly Pop Art and Abstract Expressionism, remain challenging across methods, due to the inherent ambiguity of style boundaries.

Overall, the project demonstrates the importance of deep feature learning for fine-grained visual categorization and highlights opportunities for future work, including larger datasets, style subclasses, and contextualization (artist, era, region, etc.) to boost classification accuracy.

# Dev Setup

## System Requirements
- **Operating Systems:** Windows 10+ or macOS 10.15+
- **Software Needed:**
  - Python 3.10 or higher
  - Kaggle account with API access (optional if interested in raw data)

Check installed versions:
```bash
python --version
pip --version
```

## Cloning the Repository
```bash
git clone https://github.com/kaclark219/CSCI-635-01-Group-2.git
```

## Installing Dependencies

### Virtual Environment
#### Windows
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
#### macOS/Linux
```bash
python -m venv .venv
source .venv/bin/activate
```

### Python
```bash
pip install -r requirements.txt
```

## Making a Local Development Branch
Always work on a user branch, not directly on `main`:
```bash
git fetch origin
git checkout main
git pull origin main
git checkout -b dev-<user>
```
Pull requests should be submitted to `main`.

# Dataset Overview
This project uses a subset of the **WikiArt** dataset from Kaggle, consisting of five styles:
- **Abstract Expressionism**  
- **Baroque**  
- **Cubism**  
- **Impressionism**  
- **Pop Art**

## Data Preparation Workflow
All images were processed using the pipeline in `scripts/process_images.py`, which performs:
- Resizing to **256×256**
- RGB normalization
- Removal of corrupted files
- Standardized folder structuring by class

The balanced training dataset (`data/processed_balanced/train/`) was created to reduce class imbalance, and includes augmented copies of paintings from minority classes.

## Final Dataset Splits
Located under `data/processed/`:
| Split | Description | Location |
|-------|-------------|----------|
| **Train (balanced)** | Used for CNN + embedding models | `data/processed_balanced/train/` |
| **Validation** | Used for CNN tuning + hyperparameter comparisons | `data/processed/validate/` |
| **Test** | Final evaluation set | `data/processed/test/` |

Each folder includes class-specific subdirectories:
- Abstract_Expressionism/
- Baroque/
- Cubism/
- Impressionism/
- Pop_Art/

# Project Structure Overview
```text
CSCI-635-01-Group-2/
│
├── code/
│   │
│   ├── analysis/
│   │   ├── figures/
│   │   │   └── (summary plots)
│   │   │
│   │   ├── analyze_results.ipynb
│   │   ├── confusion_matrices_summary.csv
│   │   └── model_metrics_summary.csv
│   │
│   ├── models/
│   │   ├── results/
│   │   │   ├── cnn_fast_final.weights.h5
│   │   │   └── (plots + metrics)
│   │   │
│   │   ├── cnn.ipynb
│   │   ├── dt.ipynb
│   │   ├── knn.ipynb
│   │   └── svm.ipynb
│   │
│   ├── old-models/                 # early model prototypes
│   │
│   └── scripts/
│       ├── process_images.py       # preprocessing pipeline
│       └── split_data.py           # train/val/test splitting
│ 
├── data/
│   ├── processed/
│   │   ├── test/
│   │   │   └── <class folders>
│   │   └── validate/
│   │       └── <class folders>
│   │
│   └── processed_balanced/
│       └── train/
│           └── <class folders>
│
├── resources/
│
└── README.md
```

# Running the Models
### Create Train/Validation/Test Splits (optional if using provided processed data)
```bash
python code/scripts/split_data.py
```
### Preprocess Images (optional if using provided processed data)
```bash
python code/scripts/process_images.py
```

## Running Classical Models
The classical models operate on **ResNet50 2048-dimensional embeddings**.
Open and run any of the following Jupyter notebooks:
- `code/models/knn.ipynb`
- `code/models/svm.ipynb`
- `code/models/dt.ipynb`

Each notebook will:
- Load the feature embeddings  
- Train the model  
- Generate metrics and plots  
- Save results to `code/models/results/`

## Running the CNN Model
Open:
- `code/models/cnn.ipynb`

Running all cells will:
- Load the balanced training data  
- Build and train the CNN classifier  
- Save outputs to `code/models/results/`, including:
  - `cnn_fast_final.weights.h5`  
  - `cnn_train_val_accuracy.png`  
  - `cnn_train_val_loss.png`  
  - `dataset_cnn_epoch_metrics.csv`
