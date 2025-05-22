from sklearn.pipeline import Pipeline

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
    from sklearn.svm import LinearSVC
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
    elif model_type == "svm":
        classifier = LinearSVC(C=CONFIG.default.C, dual=CONFIG.default.dual, max_iter=CONFIG.default.max_iter, random_state=42)
    elif model_type == "discriminative":
        from sklearn.feature_selection import SelectKBest, chi2
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        return Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words='english', max_features=2000)),
            ("select", SelectKBest(score_func=chi2, k=1000)),
            ("scale", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(solver='saga', multi_class='multinomial', max_iter=1000, random_state=42))
        ])
    elif model_type == "bagging":
        from src.models.config import BaggingClassifier
        return Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words='english', max_features=5000)),
            ("clf", BaggingClassifier())
        ])
    elif model_type == "boosting":
        from src.models.config import BoostingClassifier
        return Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words='english', max_features=5000)),
            ("clf", BoostingClassifier())
        ])
    else:
        raise ValueError(f"❌ Invalid model type: {model_type}.")
    
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier)
    ])
    return pipeline

def tune_hyperparameters(pipeline : Pipeline, X_train, y_train, isPCA = False):
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
    from sklearn.svm import LinearSVC
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from src.models.config import BaggingClassifier, BoostingClassifier
    def find_n_components_by_variance(X, threshold):
        """Tìm số lượng thành phần với phương sai tích lũy vượt ngưỡng (threshold)"""

        tfidf = TfidfVectorizer(lowercase=True, stop_words='english')
        X_tfidf = tfidf.fit_transform(X)  

        svd = TruncatedSVD(n_components=min(X_tfidf.shape[1], 1000)) 
        svd.fit(X_tfidf)  

        cumulative_variance = svd.explained_variance_ratio_.cumsum()

        n_components = (cumulative_variance < threshold).sum() + 1  
        return n_components
        
    classifier = pipeline.named_steps["clf"]
    
    if isinstance(classifier, DecisionTreeClassifier):
        param_grid = {
            "clf__max_depth": CONFIG.tuning.max_depth_grid,
            "clf__min_samples_split": CONFIG.tuning.min_samples_split_grid,
            "clf__ccp_alpha": CONFIG.tuning.ccp_alpha_grid
        }
    elif isinstance(classifier, MultinomialNB):
        param_grid = {"clf__alpha": CONFIG.tuning.smoothing_alpha_grid}
    elif isinstance(classifier, LinearSVC):
        param_grid = {
            "tfidf__max_features": CONFIG.tuning.tfidf__max_features,
            "tfidf__ngram_range": CONFIG.tuning.tfidf__ngram_range,
            "svm__C": CONFIG.tuning.svm__C
        }
        
        if isPCA :
            pipeline = pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(lowercase=True, stop_words='english', max_features=1000)),
                ("clf", classifier),
                ("svd", TruncatedSVD(n_components=find_n_components_by_variance(X_train, 0.95))),
            ])
    elif isinstance(classifier, LogisticRegression):
        param_grid = {
            "clf__C": CONFIG.tuning.clf__C,
            "clf__penalty": CONFIG.tuning.clf__penalty
        }
    elif isinstance(classifier, (BaggingClassifier, BoostingClassifier)): 
        return pipeline
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

# def review_data():
#     from datasets import load_from_disk
#     from src.models.config import CONFIG

#     # Load dataset
#     dataset = load_from_disk(CONFIG.processed_data_path)

#     # Kiểm tra số lượng observation
#     print(f"Train dataset size: {dataset['train'].num_rows} observations")
#     print(f"Test dataset size: {dataset['test'].num_rows} observations")
def review_data():
    """
    Display comprehensive information about the dataset including class distribution,
    text length statistics, and sample entries from each category.
    """
    from datasets import load_from_disk
    from src.models.config import CONFIG
    import numpy as np
    import pandas as pd
    from collections import Counter
    import matplotlib.pyplot as plt
    
    print("=== Dataset Review ===\n")
    
    # Load dataset
    try:
        dataset = load_from_disk(CONFIG.processed_data_path)
        print(f"✓ Successfully loaded dataset from {CONFIG.processed_data_path}\n")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Basic dataset info
    print("Dataset Overview:")
    print(f"Train dataset: {dataset['train'].num_rows:,} observations")
    print(f"Test dataset: {dataset['test'].num_rows:,} observations")
    print(f"Total dataset: {dataset['train'].num_rows + dataset['test'].num_rows:,} observations\n")
    
    # Get features and convert to pandas for easier analysis
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    # Display features/columns
    print(f"Features: {', '.join(train_df.columns.tolist())}\n")
    
    # Class distribution
    print("Class Distribution:")
    train_labels = Counter(train_df['label'])
    test_labels = Counter(test_df['label'])
    
    # Map numeric labels to categories if mapping exists
    label_map = getattr(CONFIG, 'label_map', None)
    
    for label in sorted(train_labels.keys()):
        category = label_map.get(label, label) if label_map else label
        train_count = train_labels[label]
        train_pct = 100 * train_count / train_df.shape[0]
        test_count = test_labels[label]
        test_pct = 100 * test_count / test_df.shape[0]
        
        print(f"  Category {category}:")
        print(f"    Train: {train_count:,} ({train_pct:.2f}%)")
        print(f"    Test:  {test_count:,} ({test_pct:.2f}%)")
    
    print()
    
    # Text length statistics
    print("Text Length Statistics:")
    train_df['text_length'] = train_df['text'].apply(lambda x: len(x.split()))
    
    print(f"  Min length: {train_df['text_length'].min()} words")
    print(f"  Max length: {train_df['text_length'].max()} words")
    print(f"  Mean length: {train_df['text_length'].mean():.2f} words")
    print(f"  Median length: {train_df['text_length'].median()} words")
    print()
    
    # Display sample entries from each category
    print("Sample Entries:")
    for label in sorted(train_labels.keys()):
        category = label_map.get(label, label) if label_map else label
        samples = train_df[train_df['label'] == label].sample(min(2, train_labels[label]))
        
        print(f"\n  Category: {category}")
        for i, (_, sample) in enumerate(samples.iterrows(), 1):
            print(f"    Sample {i}: \"{sample['text'][:100]}{'...' if len(sample['text']) > 100 else ''}\"")
    
    # Create visualizations directory if it doesn't exist
    import os
    if not os.path.exists('data/visualizations'):
        os.makedirs('data/visualizations')
    
    # Save class distribution plot
    plt.figure(figsize=(10, 6))
    categories = [label_map.get(label, label) if label_map else label for label in sorted(train_labels.keys())]
    train_counts = [train_labels[label] for label in sorted(train_labels.keys())]
    test_counts = [test_labels[label] for label in sorted(train_labels.keys())]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, train_counts, width, label='Train')
    plt.bar(x + width/2, test_counts, width, label='Test')
    
    plt.xlabel('Categories')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Dataset')
    plt.xticks(x, categories)
    plt.legend()
    
    plt.savefig('data/visualizations/class_distribution.png')
    print("\nClass distribution visualization saved to 'data/visualizations/class_distribution.png'")
    
    # Save text length distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(train_df['text_length'], bins=50, alpha=0.7)
    plt.xlabel('Text Length (words)')
    plt.ylabel('Frequency')
    plt.title('Text Length Distribution')
    
    plt.savefig('data/visualizations/text_length_distribution.png')
    print("Text length distribution visualization saved to 'data/visualizations/text_length_distribution.png'")
    
    # Generate a word cloud of most common words if wordcloud is installed
    try:
        from wordcloud import WordCloud
        
        # Combine all text for word cloud
        all_text = ' '.join(train_df['text'].tolist())
        
        # Generate and save word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_text)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Dataset')
        
        plt.savefig('data/visualizations/word_cloud.png')
        print("Word cloud visualization saved to 'data/visualizations/word_cloud.png'")
    except ImportError:
        print("\nNote: Install the 'wordcloud' package to generate word cloud visualizations:")
        print("pip install wordcloud")
    
    print("\n=== Dataset Review Completed ===")
    
def gen_requirement():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib
    import sklearn
    import datasets
    import pgmpy
    import hmmlearn
    # Mapping tên gói thư viện tương ứng với object module
    packages = {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "seaborn": sns.__version__,
        "matplotlib": matplotlib.__version__,
        "scikit-learn": sklearn.__version__,
        "datasets": datasets.__version__,
        "pgmpy": pgmpy.__version__,
        "hmmlearn": hmmlearn.__version__,
        # warnings và logging là built-in -> không cần ghi vào requirements.txt
    }

    # Ghi vào file requirements.txt
    with open("requirements.txt", "w") as f:
        for pkg, ver in packages.items():
            f.write(f"{pkg}=={ver}\n")

    print("✅ requirements.txt đã được tạo.")
