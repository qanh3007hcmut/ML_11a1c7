# Label
CATEGORY = ["World", "Sports", "Business", "Sci/Tech"]
# Directories
MODEL_SAVE = "models/experiments/"
PROCESSED_DATA_PATH = "data/processed/"
# Model Hyperparameters
TFIDF_MAX_FEATURES = 10000  # Number of features for TfidfVectorizer
STOP_WORDS = "english"  # Remove common English stopwords

# Naive Bayes Parameters
NB_ALPHA = [0.01, 0.1, 0.5, 1, 5, 10]  # Smoothing parameter

# Evaluation
TOP_FEATURES_TO_DISPLAY = 10  # Number of top influential features to visualize

# Naive Bayes: 
# Handle sparse features 
# Implement smoothing 
# Feature independence analysis 
# Genetic Algorithms: 
# Feature encoding for evolution 
# Design appropriate fitness function 
# Implement mutation and crossover operators 
# Selection strategy for text features 
# Population size and generation management 
