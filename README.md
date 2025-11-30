# CSCI-635-01-Group-2
**An exploration of classifying art styles of famous works using various ML models.** <br /> <br />
Created for CSCI-635 (Introduction to Machine Learning) at RIT fall 2025, this project explores various supervised and unsupevised models for classification of art styles when given pictoral data of a painting.


## Table of Contents
- [Project Overiew](#CSCI-635-01-Group-2)
- [Table of Contents](#table-of-contents)
- [Abstract](#abstract)
- [Dev Setup](#dev-setup)
  - [System Requirements](#system-requirements)
  - [Making a Local Development Branch](#making-a-local-development-branch)
  - [Cloning the Repository](#cloning-the-repository)
  - [Installing Dependencies](#installing-dependencies)

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

## Making a Local Development Branch
Always work on a user branch, not directly on `main`:
```bash
git fetch origin
git checkout main
git pull origin main
git checkout -b dev-<user>
```
Pull requests should be submitted to `main`.

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