# I-Comprehend
 Thesis Project | Laguna State Polytechnic University - Computer Science

## Overview
This project aims to design, develop, implement, and evaluate a
system with a multimodal approach, incorporating quiet reading, loud reading, question
generation, and advanced recognition technologies using machine learning (ML) approaches
to enhance college students’ reading comprehension.

## Prerequisites
Before you begin, ensure you have the following prerequisites:
- Version of Python (Python 3.12.3)
- Virtual environment
- Required Python Packages (listed in `requirements.txt`)

## Installation
You can download the necessary files from this Google Drive Link.

[Google Drive Link](https://drive.google.com/drive/folders/1vvXvlYrBWtJxfcqEK0a4BerMbsno3lcx?usp=sharing) (t5_model, t5_tokenizer, qg_model, qgtokenizer)

after downloading make or move the model to models directory 

## Install Dependencies

- before installing the dependencies create a virtual environment using virtualenv
to pip install virtual env
```bash
pip install virtualenv
```
to create a virtual environment
```bash
virtualenv env
```
to activate environment
```bash
env\Scripts\activate
```

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt

```

# Project Directory Structure

This guide provides a detailed overview of the project directory structure, explaining the purpose and typical content of each folder. Follow this guide to understand the organization of the files and how to use them effectively.

## 1. Root Directory (`/`)

- **Purpose:** Main directory containing top-level files and scripts.
- **Typical Content:**
  - **`README.md`**: Provides an overview of the project, setup instructions, usage examples, and links to further documentation.
  - **`.gitignore`**: Specifies files and directories to be ignored by Git (e.g., `datasets/`, `logs/`).
  - **`requirements.txt` or `environment.yml`**: Lists dependencies for the project, which can be installed using pip or conda.
  - **`pyrightconfig.json`**: Configuration file for Pyright, specifying type-checking settings.

## 2. `src/` (Source Code)

- **Purpose:** Houses all source code files related to the project.
- **Subfolders & Content:**
  - **`models/`**:
    - Contains model architecture definitions and related classes.
    - **Files:**
      - `t5_models`: T5 model implementation with custom layers and configurations.
  - **`data/`**:
    - Scripts for data loading, preprocessing, and augmentation.
    - **Files:**
      - `data_loader.py`: Functions for reading and splitting data into training, validation, and test sets.
      - `preprocessing.py`: Includes data cleaning, feature engineering, and augmentation steps.
  - **`utils/`**:
    - Utility scripts for general-purpose functions.
    - **Files:**
      - `metrics.py`: Custom evaluation metrics and loss functions.
      - `plotting.py`: Functions for plotting training curves, confusion matrices, etc.

## 3. `datasets/`

- **Purpose:** Stores datasets used for training and evaluation.
- **Subfolders & Content:**
  - **`raw/`**:
    - Unmodified original data directly from the source.
    - **Files:**
      - `train.csv`: Raw training data file.
      - `test.json`: Raw test data in JSON format.
  - **`processed/`**:
    - Preprocessed data files ready for model consumption.
    - **Files:**
      - `train_processed.pkl`: Pickle file containing processed training data.
      - `test_processed.pkl`: Pickle file for processed test data.

## 4. `notebooks/`

- **Purpose:** Contains Jupyter notebooks for data exploration, analysis, and experimental prototyping.
- **Typical Content:**
  - `EDA_.ipynb`: Notebook performing exploratory data analysis (EDA).
  - `model_prototyping.ipynb`: Used for model experimentation, hyperparameter tuning, and initial testing.

## 5. `experiments/`

- **Purpose:** Stores logs, configurations, and checkpoints from various experiment runs.
- **Subfolders & Content:**
  - **`logs/`**:
    - Logs from training sessions, including loss values, accuracy, and other metrics.
    - **Files:**
      - `training_log.txt`: Text log of training progress.
      - `tensorboard/`: TensorBoard log files for visualizing metrics during training.
  - **`checkpoints/`**:
    - Model weights and checkpoints saved during training for easy restoration.
    - **Files:**
      - `model_epoch_10.h5`: Model weights saved after the 10th epoch.
      - `best_model.pth`: Best-performing model checkpoint.

## 6. `outputs/`

- **Purpose:** Stores the final outputs, such as generated plots, evaluation results, and predictions.
- **Subfolders & Content:**
  - **`plots/`**:
    - Visualizations generated from analysis or during training.
    - **Files:**
      - `accuracy_plot.png`: Plot showing accuracy trends over epochs.
      - `confusion_matrix.png`: Confusion matrix for model evaluation.
  - **`predictions/`**:
    - Stores model predictions on test data or other datasets.
    - **Files:**
      - `test_predictions.csv`: CSV file containing model predictions on the test set.
      - `generated_questions.json`: JSON file with predictions from the model.

## 7. `configs/`

- **Purpose:** Holds configuration files to manage settings for experiments, models, and data paths.
- **Typical Content:**
  - `config.yaml`: Main configuration file with adjustable parameters like learning rate, batch size, and file paths.
  - `model_config.json`: Specific configuration settings for model hyperparameters.

## 8. `tests/`

- **Purpose:** Contains tests for ensuring the code's correctness and robustness.
- **Typical Content:**
  - `test_models.py`: Unit tests for validating model architectures and outputs.
  - `test_data.py`: Tests for data loading and preprocessing functions to ensure expected outputs.

## 9. `scripts/`

- **Purpose:** Standalone scripts for specific tasks, such as data preparation, downloading, or miscellaneous utilities.
- **Typical Content:**
  - `download_data.py`: Script to fetch data from an external source, like an API or a public dataset.
  - `clean_data.py`: Script for cleaning and formatting raw data files.

## 10. Other Files

- **`.gitignore`:** Include file extensions and folder names that should not be tracked (e.g., `*.log`, `__pycache__/`, `*.h5`).
- **`LICENSE`:** (Optional) Defines the terms under which your project can be used or distributed.

---

## Run the application
To run the server, run the following command (go to the root directory first):
```bash
python main.py
```

## Run the evaluation
To run the evaluation, run the following command (go to the root directory first):
```bash
python evaluation.py
```
