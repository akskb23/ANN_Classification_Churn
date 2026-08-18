# ANN Classification — Customer Churn

This repository contains an implementation of an Artificial Neural Network (ANN) for binary classification of customer churn. The goal of the project is to predict whether a customer will churn (leave) based on features in the dataset and to provide a reproducible pipeline for preprocessing, training, evaluation, and inference.
## Table of contents
- Project overview
- What was implemented
- Data
- Preprocessing
- Model (ANN) architecture
- Training & evaluation
- Results
- How to run
- Files and structure
- Dependencies
- Reproducibility & hyperparameters
- Contributing
- License

## Project overview
This project demonstrates a complete pipeline for training an ANN to predict customer churn. The pipeline includes data loading, cleaning/preprocessing, feature engineering (if applicable), training an ANN classifier, evaluating performance with common metrics, and saving the trained model for inference.

## What was implemented
- Data ingestion pipeline for the provided churn dataset
- Preprocessing steps such as handling missing values, encoding categorical variables, scaling numeric features, and splitting into train/validation/test sets
- An ANN built with a high-level framework (Keras / TensorFlow recommended) with configurable layers, activations, and regularization
- Training loop with callbacks (early stopping and model checkpointing)
- Evaluation using accuracy, precision, recall, F1-score, ROC AUC, and a confusion matrix
- Scripts or notebooks for inference and generating evaluation plots (training curves, ROC)

## Data
- Expected to find a dataset (CSV) in a data/ directory (e.g., data/churn.csv). If a different location or filename was used, update the config accordingly.
- The dataset should contain a binary target column (e.g., `churn`, `Exited`, or similar). Adjust the target name in training scripts if different.

## Preprocessing
Typical preprocessing steps implemented or recommended:
- Drop/handle missing values (drop rows or impute)
- Encode categorical features (one-hot or ordinal encoding as appropriate)
- Scale numeric features (StandardScaler or MinMaxScaler)
- Split dataset into training/validation/test sets (recommended split: 70/15/15 or similar)

## Model (ANN) architecture
A representative ANN used in this project:
- Input layer matching the number of input features
- 1–3 hidden layers (example: Dense 64 -> Dense 32) with ReLU activation
- Dropout and/or L2 regularization to reduce overfitting
- Output layer with a single unit and sigmoid activation for binary classification
- Binary cross-entropy loss and an optimizer such as Adam

Hyperparameters (example defaults):
- Batch size: 32
- Epochs: 100 with EarlyStopping (patience 10)
- Learning rate: 1e-3

Adjustable in config or training script.

## Training & evaluation
- Training uses training set, with a validation split for early stopping
- Model saved to `models/` (for example `models/ann_churn.h5` or SavedModel format)
- Evaluation script computes:
  - Accuracy
  - Precision, recall, F1-score
  - ROC AUC
  - Confusion matrix and classification report
- Visualizations: training/validation loss and accuracy curves, ROC curve

## Results
- Reported metrics (example placeholders — replace with actual results after running training):
  - Accuracy: 0.85
  - Precision: 0.78
  - Recall: 0.72
  - F1-score: 0.75
  - ROC AUC: 0.88

Include your actual numbers after running the training pipeline.

## How to run
1. Create and activate a Python environment (recommended):
   - python -m venv .venv
   - On Windows: .\\.venv\\Scripts\\Activate.ps1 (PowerShell) or .\\.venv\\Scripts\\activate (cmd)
2. Install required packages:
   - pip install -r requirements.txt
   If requirements.txt is not present, common packages include:
   - pip install numpy pandas scikit-learn matplotlib seaborn tensorflow keras joblib
3. Prepare the dataset: place the CSV in `data/` (e.g., `data/churn.csv`) or update the path in the training script.
4. Run training (example):
   - python src/train.py --data-path data/churn.csv --output-dir models --epochs 100
5. Evaluate (example):
   - python src/evaluate.py --model models/ann_churn.h5 --data-path data/churn.csv
6. Inference (example):
   - python src/predict.py --model models/ann_churn.h5 --input data/sample_input.csv --output predictions.csv

Adapt argument names to the actual scripts provided in this repo.

## Files and structure
A typical project structure for this repository:

- data/
  - churn.csv (dataset)
- src/
  - train.py (training script)
  - evaluate.py (evaluation script)
  - predict.py (inference script)
  - preprocess.py (data preprocessing functions)
  - model.py (model building function)
- models/
  - ann_churn.h5 (trained model)
- notebooks/
  - analysis.ipynb (exploratory analysis and experiments)
- requirements.txt
- README.md

If files in this repo use different names or locations, update the structure section accordingly.

## Dependencies
Core dependencies (examples):
- Python 3.8+
- numpy
- pandas
- scikit-learn
- tensorflow (or tensorflow-cpu)
- matplotlib
- seaborn
- joblib

If a requirements.txt exists, prefer installing from it.

## Reproducibility & hyperparameters
- Use fixed random seeds for numpy, TensorFlow and any other RNGs for reproducibility
- Save preprocessing objects (scalers/encoders) with joblib or pickle so they can be reused at inference time
- Save training metrics and the model checkpoint


