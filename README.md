# Text Classification Pipeline

This project implements a machine learning pipeline for text classification using the AG News dataset. The pipeline includes data preprocessing, model training, evaluation, and prediction capabilities.

## Overview

The system supports multiple text classification models:
- Naive Bayes
- Decision Tree
- Neural Network (MLP)
- Bayesian Network
- Hidden Markov Model

Each model is designed to categorize news articles into one of four categories: World, Sports, Business, and Sci/Tech.

## Getting Started

### Prerequisites

This project requires Python 3.7+ and several dependencies which are listed in the `requirements.txt` file.

You can install all dependencies with:

```bash
pip install -r requirements.txt
```

Alternatively, the main libraries used include:
- scikit-learn
- pandas
- numpy
- datasets
- matplotlib
- seaborn
- pgmpy (for Bayesian Network)
- hmmlearn (for Hidden Markov Model)

## Usage

The main entry point for the pipeline is `run.py`, which provides a command-line interface to interact with the system.

### Command Line Interface

```
python run.py [OPTIONS]
```

### Available Options

#### Data Processing

```bash
# Preprocess the dataset
python run.py --task preprocess

# Review the processed dataset
python run.py --task review
```

#### Model Training

```bash
# Train a model (replace MODEL_NAME with one of the available models)
python run.py --train MODEL_NAME
```

Available models:
- `naive_bayes`
- `decision_tree`
- `neural_network`
- `bayesian_network`
- `hidden_markov_model`

Example:
```bash
python run.py --train naive_bayes
```

#### Making Predictions

```bash
# Make predictions using a trained model
python run.py --predict MODEL_NAME
```

Example:
```bash
python run.py --predict decision_tree
```

#### Testing Models

```bash
# Test a model on sample data
python run.py --task test MODEL_NAME
```

Example:
```bash
python run.py --task test neural_network
```

## Pipeline Structure

1. **Data Preprocessing**
   - Loads AG News dataset
   - Maps numerical labels to categories (World, Sports, Business, Sci/Tech)
   - Saves processed data to disk

2. **Feature Engineering**
   - Text vectorization using TF-IDF or Count Vectorizers
   - Feature selection

3. **Model Training**
   - Hyperparameter tuning using grid search
   - Model training with optimal parameters
   - Model serialization for later use

4. **Evaluation & Prediction**
   - Accuracy, precision, recall, F1-score metrics
   - Confusion matrix visualization
   - Cross-validation reporting
   - ROC-AUC for applicable models

## Project Structure

```
.
├── data/
│   ├── raw/                # Raw dataset files
│   └── processed/          # Preprocessed dataset files
├── models/
│   ├── experiments/        # Experimental model files
│   └── trained/            # Final trained models
├── src/
│   ├── data/               # Data loading and preprocessing
│   ├── features/           # Feature engineering and utilities
│   ├── models/             # Model definitions and training
│   └── visualization/      # Visualization tools
├── tests/                  # Test scripts and sample data
├── requirements.txt        # Project dependencies
└── run.py                  # Main pipeline interface
```

## Example Workflow

A typical workflow using this pipeline:

1. **Use a preprocessed dataset (stored in `data/processed/`)**:
   ```bash
   python run.py --task review
   ```

2. **Use a pre-trained model (stored in `models/experiments/` or `models/trained/`) for prediction and evaluation**:
   ```bash
   python run.py --predict naive_bayes
   ```

3. **Test the model on custom examples**:
   ```bash
   python run.py --task test naive_bayes
   ```

4. **If training a new model is needed, it will be stored in `models/experiments/`**:
   ```bash
   python run.py --train naive_bayes
   ```

## Performance Monitoring

The pipeline includes a timer utility that logs execution time for training and prediction tasks, helping to monitor model efficiency.

## Additional Notes

- Models are serialized using joblib for efficient storage and loading
- The system includes cross-validation to ensure model robustness
- Confusion matrices help visualize classification performance
