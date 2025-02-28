def convert_to_sparse(data, stopWords, maxFeatures, ngram_range=(1,1)):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words=stopWords, max_features=maxFeatures, ngram_range=ngram_range)
    X = vectorizer.fit_transform(data)
    return X

def get_dataset():
    from datasets import load_from_disk
    DATA_PATH = "data/processed/"
    processed_dataset = load_from_disk(DATA_PATH)
    print(f"Train dataset size: {processed_dataset['train'].num_rows} observations")
    return processed_dataset

def load_trained_model(model_name):
    """Load the trained model from a .pkl file and return it for prediction."""
    import os
    import joblib

    from src.models.config import MODEL_SAVE
    
    model_path  = os.path.join(MODEL_SAVE, f"{model_name}.pkl")
    if not os.path.exists(model_path ):
        raise FileNotFoundError(f"❌ Model file not found at {model_path }. Train and save the model first.")

    model = joblib.load(model_path )
    print(f"✅ Model loaded successfully from {model_path }")
    return model

def save_trained_model(model, model_name):
    import os
    import joblib
    from src.models.config import MODEL_SAVE
    model_path  = os.path.join(MODEL_SAVE, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Model saved at: {model_path}")

def review_data():
    from datasets import load_from_disk
    from src.models.config import PROCESSED_DATA_PATH

    # Load dataset
    DATA_PATH = PROCESSED_DATA_PATH
    dataset = load_from_disk(DATA_PATH)

    # Kiểm tra số lượng observation
    print(f"Train dataset size: {dataset['train'].num_rows} observations")
    print(f"Test dataset size: {dataset['test'].num_rows} observations")

def get_best_alpha(X_train, train_labels):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import GridSearchCV
    from src.models.config import NB_ALPHA
    # Grid Search để tìm giá trị alpha tối ưu
    param_grid = {'alpha': NB_ALPHA}
    grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, train_labels)
    # Lấy giá trị alpha tốt nhất
    best_alpha = grid_search.best_params_['alpha']
    print(f"🔥 Best alpha found: {best_alpha}")
    return best_alpha