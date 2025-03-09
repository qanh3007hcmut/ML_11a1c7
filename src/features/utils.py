def build_pipeline(model_type):
    """
    Builds a text classification pipeline with TF-IDF vectorizer and Decision Tree classifier.

    Args:
        model_type (str): The type of model to use. Supported values are:
            - "decision_tree": Uses DecisionTreeClassifier.
            - "naive_bayes": Uses MultinomialNB.

    Raises:
        ValueError: If an invalid model type is provided.

    Returns:
        pipeline (Pipeline): A scikit-learn pipeline.
    """ 
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import MultinomialNB
    from src.models.config import CONFIG
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=1000)

    if model_type == "decision_tree":
        classifier = DecisionTreeClassifier(max_depth=CONFIG.default.max_depth,
                                            min_samples_split=CONFIG.default.min_samples_split,
                                            ccp_alpha=CONFIG.default.ccp_alpha,
                                            random_state=42)
    elif model_type == "naive_bayes":
        classifier = MultinomialNB(alpha=CONFIG.default.smoothing_alpha)
    
    elif model_type == "neural_network":
        from src.models.config import MLPTextClassifier
        return MLPTextClassifier()
    
    elif model_type == "bayesian_network":
        from src.models.config import BayesianNetworkClassifier
        return BayesianNetworkClassifier()

    elif model_type == "hidden_markov_model":
        from src.models.config import HMMClassifier
        return HMMClassifier()
    
    else:
        raise ValueError(f"❌ Invalid model type: {model_type}.")
    
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier)
    ])
    return pipeline

def tune_hyperparameters(pipeline, X_train, y_train):
    """Tunes hyperparameters for Decision Tree or Naive Bayes classifier.
    
    Args:
        pipeline: Sklearn pipeline containing either DecisionTreeClassifier or MultinomialNB.
        X_train: Training data features.
        y_train: Training data labels.
    
    Returns:
        The best estimator (trained model with optimal hyperparameters).
    """
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from src.models.config import CONFIG
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import MultinomialNB

    classifier = pipeline.named_steps["clf"]
    
    if isinstance(classifier, DecisionTreeClassifier):
        param_grid = {
            "clf__max_depth": CONFIG.tuning.max_depth_grid,
            "clf__min_samples_split": CONFIG.tuning.min_samples_split_grid,
            "clf__ccp_alpha": CONFIG.tuning.ccp_alpha_grid
        }
    elif isinstance(classifier, MultinomialNB):
        param_grid = {"clf__alpha": CONFIG.tuning.smoothing_alpha_grid}
    else:
        raise ValueError("❌ Unsupported model type in pipeline!")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print(f"✅ Best hyperparameters found: {grid_search.best_params_}")

    return grid_search.best_estimator_  

def get_dataset():
    from datasets import load_from_disk
    from src.models.config import CONFIG
    processed_dataset = load_from_disk(CONFIG.processed_data_path)
    print(f"Train dataset size: {processed_dataset['train'].num_rows} observations")
    return processed_dataset

def load_trained_model(model_name):
    """Load the trained model from a .pkl file. Try joblib first, then fall back to pickle if needed."""
    import os
    import joblib
    import pickle

    from src.models.config import CONFIG
    
    model_path = os.path.join(CONFIG.model_save, f"{model_name}.pkl")
    trained_path = os.path.join(CONFIG.trained_path, f"{model_name}.pkl")
    
    if not os.path.exists(trained_path):
        print(f"❌ Verified trained model not found at {trained_path}. Loading from experiments {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found at {model_path}. Train and save the model first.")
    else: 
        model_path = trained_path
        
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully using joblib from {model_path}")
    except Exception as e:
        print(f"⚠️ Joblib load failed: {e}. Trying with pickle...")
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print(f"✅ Model loaded successfully using pickle from {model_path}")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model using both joblib and pickle: {e}")

    return model

def save_trained_model(model, model_name):
    """Save the trained model to a .pkl file."""
    import os
    import joblib
    from src.models.config import CONFIG
    model_path  = os.path.join(CONFIG.model_save, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Model saved at: {model_path}")

def review_data():
    from datasets import load_from_disk
    from src.models.config import CONFIG

    # Load dataset
    dataset = load_from_disk(CONFIG.processed_data_path)

    # Kiểm tra số lượng observation
    print(f"Train dataset size: {dataset['train'].num_rows} observations")
    print(f"Test dataset size: {dataset['test'].num_rows} observations")