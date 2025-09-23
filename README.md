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
  - [Downloading the Dataset](#downloading-the-dataset)

---
# Abstract
Abstract will be pasted here once completed.

# Dev Setup

## System Requirements
- **Operating Systems:** Windows 10+ or macOS 10.15+
- **Software Needed:**
  - Python 3.10 or higher
  - Kaggle account with API access

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

### Kaggle API
#### Windows
```bash
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle" | Out-Null
Move-Item -Path ".\kaggle.json" -Destination "$env:USERPROFILE\.kaggle\kaggle.json"
```
#### macOS/Linux
```bash
mkdir -p ~/.kaggle
mv ./kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Downloading the Dataset
WARNING: This will take a WHILE! There is a lot of data, so this step could take about 30 minutes to an hour.
```bash
mkdir -p data/raw
kaggle datasets download -d steubk/wikiart -p data/raw --unzip
```
Then unzip the folder within the data/raw folder, and run the following script to split the data; this step may also take a while.
```bash
python code/scripts/split_data.py
```