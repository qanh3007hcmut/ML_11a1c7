# Text Classification Pipeline

A machine learning pipeline for classifying news articles into four categories (World, Sports, Business, Sci/Tech) using the [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news?row=2&fbclid=IwY2xjawI6YoNleHRuA2FlbQIxMAABHSTM_lg9XiOtTpdj_7S_7xlHy-WQqF1ljeKtApc8FyCYgbTjFnRpT3S0Tg_aem_vtQpfcUcv-ZrxQfPN-94yQ) dataset. 

The **AG News** dataset serves as a robust benchmark for text classification, comprising thousands of news articles sourced from reputable outlets such as **Reuters** and **AP**.

The pipeline includes data preprocessing, model training, evaluation, and prediction capabilities.

## Quick Start
```bash
pip install -r requirements.txt
python run.py --task preprocess
python run.py --train naive_bayes
python run.py --predict naive_bayes
```

## Supported Models

- **Naive Bayes** (`naive_bayes`) - Fast baseline classifier
- **Decision Tree** (`decision_tree`) - Interpretable model
- **Neural Network** (`neural_network`) - MLP for complex patterns
- **Bayesian Network** (`bayesian_network`) - Probabilistic graphical model
- **Hidden Markov Model** (`hidden_markov_model`) - Sequential text classifier

## Command Reference
The main entry point for the pipeline is `run.py`, which provides a command-line interface to interact with the system.
```bash
python run.py [OPTIONS]
```
### Data Operations
- **Preprocess raw data**

```bash
python run.py --task preprocess
```

- **View dataset statistics**
```bash
python run.py --task review
```

### Model Operations
- **Train model** (saves to models/experiments/)
```bash
python run.py --train MODEL_NAME
```
- **Predict and Evaluate** model on test set
```bash
python run.py --predict MODEL_NAME
```
- **Test** model on custom examples
```bash
python run.py --test MODEL_NAME
```
## Pipeline Components

1. **Data Processing**
   - Loads AG News dataset
   - Maps labels to categories
   - Handles text normalization

2. **Feature Engineering**
   - Text vectorization (TF-IDF/Count)
   - Feature selection optimization

3. **Model Training**
   - Hyperparameter tuning
   - Cross-validation
   - Model serialization

4. **Evaluation**
   - Metric: accuracy, precision, recall, F1-score 
   - Confusion matrix visualization
   - Cross-validation reporting

## Performance Monitoring

The pipeline includes a timer utility that logs execution time for training and prediction tasks, helping to monitor model efficiency.

## Additional Notes

- Models are serialized using joblib for efficient storage and loading
- The system includes cross-validation to ensure model robustness
- Confusion matrices help visualize classification performance
 
## Example Workflow

### Training a New Model

```bash
# Process raw data (if not already done)
python run.py --task preprocess

# Train Naive Bayes model
python run.py --train naive_bayes
```
#### Expected Output
```bash
Data saved at: data/processed/
Train dataset size: 120000 observations
Best hyperparameters found: {'clf__alpha': 0.1}
Training finished in 117 seconds (1.95 minutes)
Model saved at: models/experiments/naive_bayes.pkl
```
### Using a Pre-trained Model

```bash
# Make predictions on test data
python run.py --predict naive_bayes
```
#### Expected Output
```bash
Train dataset size: 120000 observations
Model loaded successfully using joblib from models/trained/naive_bayes.pkl
Predicting finished in 0 seconds (0.00 minutes)

=== Test Set Performance ===
Accuracy: 0.8525
Precision: 0.8516
Recall: 0.8525
F1-score: 0.8518

Classification Report:
               precision    recall  f1-score   support
       World       0.82      0.80      0.81      1900
      Sports       0.82      0.81      0.82      1900
    Business       0.90      0.94      0.92      1900
    Sci/Tech       0.86      0.87      0.87      1900

    accuracy                           0.85      7600
   macro avg       0.85      0.85      0.85      7600
weighted avg       0.85      0.85      0.85      7600
```

### Test with custom data saved at tests/test_data.py

```bash
python run.py --test naive_bayes
# "Stock markets are seeing a huge drop today."
# "The football team won the championship!"
# "NASA discovered a new exoplanet in space."
```
#### Expected Output
```bash
🛠 Testing the model...
Model loaded successfully using joblib from models/trained/naive_bayes.pkl
Testing finished in 0 seconds (0.00 minutes)
Predicted Categories: ['Business', 'Sports', 'Sci/Tech']
```
