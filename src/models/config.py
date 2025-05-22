from dataclasses import dataclass, field
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.ensemble import BaggingClassifier as Bagging
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from hmmlearn import hmm
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class DefaultConfig:
    # Default Model Parameters
    smoothing_alpha: float = 0.1 # Laplace Smoothing
    max_depth: Optional[int] = 10
    min_samples_split: int = 5
    ccp_alpha: float = 0.0
    C: float = 1.0
    dual: bool = False
    max_iter: int = 5000

@dataclass(frozen=True)
class TuningConfig:
    # Grid Search Parameters
    smoothing_alpha_grid: List[float] = field(default_factory=lambda: [0.01, 0.1, 0.5, 1, 5, 10])
    max_depth_grid: List[Optional[int]] = field(default_factory=lambda: [10, 20, None])
    min_samples_split_grid: List[int] = field(default_factory=lambda: [2, 5, 10])
    ccp_alpha_grid: List[float] = field(default_factory=lambda: [0.0, 0.001, 0.01])
    tfidf__max_features: List[int] = field(default_factory=lambda: [10000, 20000])
    tfidf__ngram_range: List[tuple[int]] = field(default_factory=lambda: [(1, 1),(1, 2)]),
    svm__C: List[float] = field(default_factory=lambda: [0.1, 1.0])
    clf__C: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    clf__penalty: int = field(default_factory=lambda: ['l2'])
@dataclass(frozen=True)
class Config:
    # Model
    model_dict = {
        "naive_bayes": "Naive Bayes",
        "decision_tree": "Decision Tree",
        "neural_network": "Neural Network Model",
        "bayesian_network": "Bayesian Network",
        "hidden_markov_model": "Hidden Markov Model",
        "svm" : "SVM",
        "svm_pca" : "SVM & PCA",
        "discriminative" : "Discriminative Model",
        "bagging" : "Bagging Classifier",
        "boosting" : "Boosting Classifier",
        "bagging_model" : "Bagging Classifier",
        "boosting_model" : "Boosting Classifier"
    }
    
    # Labels
    CATEGORY_MAPPING = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    categories: List[str] = field(default_factory=lambda: ["World", "Sports", "Business", "Sci/Tech"])

    # Directories
    model_save: str = "models/experiments/"
    trained_path: str = "models/trained/"
    processed_data_path: str = "data/processed/"
    
    # TF-IDF Parameters
    tfidf_max_features: int = 10000
    stop_words: str = "english"

    # Evaluation
    top_features_to_display: int = 10
    
    default: DefaultConfig = DefaultConfig()
    tuning: TuningConfig = TuningConfig()

# Instantiate config object
CONFIG = Config()

class MLPTextClassifier:
    def __init__(self, hidden_layer_sizes=(64, 32), alpha=0.001, max_iter=100):
        """
        Initialize the MLPTextClassifier.

        Parameters:
        hidden_layer_sizes (tuple): The ith element represents the number of neurons in the ith hidden layer.
        alpha (float): L2 penalty (regularization term) parameter.
        max_iter (int): Maximum number of iterations.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=1000)
        self.label_encoder = LabelEncoder()
        self.model = None
    
    def fit(self, X_train, y_train):
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        self.model = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, activation='relu', solver='adam',
                                   alpha=self.alpha, random_state=42, batch_size=32, learning_rate_init=0.01,
                                   max_iter=self.max_iter)
        self.model.fit(X_train_tfidf, y_train_encoded)
    
    def predict(self, X_test):
        X_test_tfidf = self.vectorizer.transform(X_test)
        predictions = self.model.predict(X_test_tfidf)
        return self.label_encoder.inverse_transform(predictions)

class BayesianNetworkClassifier:
    def __init__(self, n_bins=10, max_features=1000):
        # Set max_features to capture all the feature
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
        
        # Discretizer to make discrete value from vector
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')

        # Label encoder to encode y column to discrete value
        self.label_encoder = LabelEncoder()
        self.model = None

    def fit(self, X_train, y_train):
        """
        Fit the Bayesian Network with given X_train and y_train data.
        """
        # Vectorize the text (or X column)
        X_tfidf = self.vectorizer.fit_transform(X_train).toarray()

        # Discretization
        X_disc = self.discretizer.fit_transform(X_tfidf)
        
        # Encode the labels to discrete integer
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        # Each word depends on a word with specific discrete value (or label)
        df = pd.DataFrame(X_disc, columns=[f'word_{i}' for i in range(X_disc.shape[1])])
        df['Category'] = y_encoded
        
        # word ith will depend on the correspond Category due to Naive Bayes
        structure = [('Category', f'word_{i}') for i in range(X_disc.shape[1])]
        self.model = BayesianNetwork(structure)
        
        # fitting with Maximum Likelihood Estimator
        self.model.fit(df, estimator=MaximumLikelihoodEstimator)
    
    def predict(self, X_test):
        """
        Get model prediction for all the data in X_test.
        """
        # transform a string to vector and discretized value before
        X_tfidf = self.vectorizer.transform(X_test).toarray()
        X_disc = self.discretizer.transform(X_tfidf)

        # eliminate unnecessary variable to cut down the computation
        inference = VariableElimination(self.model)

        # fitting process
        predictions = []
        for row in X_disc:
            evidence = {f'word_{i}': int(row[i]) for i in range(len(row))}
            try:
                result = inference.map_query(variables=['Category'], evidence=evidence, show_progress=False)
                predictions.append(int(result['Category']))
            except Exception as e:
                predictions.append(0)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X_test):
        """
        Probability of each prediction for evaluation.
        """
        # Testing process
        X_tfidf = self.vectorizer.transform(X_test).toarray()
        X_disc = self.discretizer.transform(X_tfidf)
        inference = VariableElimination(self.model)

        # Store probabilities
        prob_predictions = []
        for row in X_disc:
            evidence = {f'word_{i}': int(row[i]) for i in range(len(row))}
            try:
                result = inference.query(variables=['Category'], evidence=evidence, show_progress=False)
                prob_predictions.append(result.values)
            except Exception as e:
                prob_predictions.append(np.zeros(len(self.label_encoder.classes_)))
        return np.array(prob_predictions)

class HMMClassifier:
    def __init__(self, n_components=5, covariance_type='diag', n_iter=100, max_features=1000):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.max_features = max_features
        self.category_hmms = {}
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        self.label_encoder = LabelEncoder()

    def fit(self, X_train, y_train):
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        X_train_counts = self.vectorizer.fit_transform(X_train).toarray()
        
        for category in self.label_encoder.classes_:
            idx = (y_train == category)
            model = hmm.GaussianHMM(n_components=self.n_components, covariance_type=self.covariance_type, n_iter=self.n_iter)
            model.fit(X_train_counts[idx])
            self.category_hmms[category] = model

    def predict(self, X):
        X_counts = self.vectorizer.transform(X).toarray()
        predictions = []
        for sample in X_counts:
            scores = {cat: model.score(sample.reshape(1, -1)) for cat, model in self.category_hmms.items()}
            predicted_category = max(scores, key=scores.get)
            predictions.append(predicted_category)
        return np.array(predictions)

class BoostingClassifier:
    def __init__(self, n_estimators=10, random_state=42):
        self.base_models = [
            ('lr', LogisticRegression(max_iter=1000)),
            ('nb', MultinomialNB()),
            ('dt', DecisionTreeClassifier(max_depth=100))
        ]
        self.voting_clf = VotingClassifier(estimators=self.base_models, voting='soft', weights=[2, 1, 1])
        self.model = AdaBoostClassifier(estimator=self.voting_clf, n_estimators=n_estimators, random_state=random_state)
        
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class BaggingClassifier:
    def __init__(self, n_estimators=50, random_state=42):
        self.base_models = [
            ('lr', LogisticRegression(max_iter=1000)),
            ('nb', MultinomialNB()),
            ('dt', DecisionTreeClassifier(max_depth=10))
        ]
        self.voting_clf = VotingClassifier(estimators=self.base_models, voting='soft')
        self.model = Bagging(estimator=self.voting_clf, n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)