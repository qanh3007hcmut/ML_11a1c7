import argparse
from src.data.preprocess import load_and_preprocess_data
from src.features.evaluation import *
from src.features.ultils import get_dataset
from src.models.config import BayesianNetworkClassifier, HMMClassifier, MLPTextClassifier
import warnings
warnings.simplefilter("ignore", category=UserWarning)

def main():
    parser = argparse.ArgumentParser(description="Run various parts of the ML pipeline.")
    usage_text = """
    Usage Guide for run.py
    =======================
    
    This script allows you to preprocess data, train models, and make predictions using various machine learning models.
    
    Commands:
    ---------
    
    1. Preprocess Dataset:
       python run.py --task preprocess
       
    2. Review Dataset:
       python run.py --task review
       
    3. Train a Model:
       python run.py --train <model_name>
       
       Available models:
       - naive_bayes
       - decision_tree
       - neural_network
       - bayesian_network
       - hidden_markov_model
       
       Example:
       python run.py --train naive_bayes
       
    4. Predict Using a Model:
       python run.py --predict <model_name>
       
       Example:
       python run.py --predict decision_tree
       
    Notes:
    ------
    - Ensure the dataset is preprocessed before training or predicting.
    - Models must be trained before making predictions.
    
    """
    # Add pre action  
    parser.add_argument(
        "--task", 
        type=str, 
        nargs="+",
        help="Task to execute: fetch, save, review data and test model"
    )
    
    # Add train action with choice of model
    parser.add_argument(
        "--train", 
        type=str, 
        choices=["naive_bayes", "decision_tree", "neural_network", "bayesian_network", "hidden_markov_model"],
        help="choose model to train with dataset ag_news"
    )
    
    # Add predict action with choice of model
    parser.add_argument(
        "--predict", 
        type=str, 
        choices=["naive_bayes", "decision_tree", "neural_network", "bayesian_network", "hidden_markov_model"],
        help="choose model to predict with dataset ag_news test"
    )
    
    args = parser.parse_args()

    if args.task:
        if not args.task[0]:
            parser.error("No task specified. Use --task preprocess, review, or test <model_name>")
            
        elif args.task[0] == "preprocess":
            print("📥 Fetching and preprocess dataset...")
            load_and_preprocess_data()
        
        elif args.task[0] == "review":
            print("Reviewing dataset...")
            get_dataset()
        elif args.task[0] == "test":
            from tests.test_models import test_model_classification
            if len(args.task) < 2:
                parser.error("You must specify a model name after --task test")
            
            model_name = args.task[1]
            valid_models = ["naive_bayes", "decision_tree", "neural_network", "bayesian_network", "hidden_markov_model"]
            if model_name not in valid_models:
                parser.error(f"Invalid model name. Choose from: {valid_models}")
            
            print("🛠 Testing the model...")
            test_model_classification(model_name)
    
    elif args.train:    
        from src.models.train_model import train_model_classifiers as train_model
        from src.models.config import CONFIG
        print(f"↗️  Training {CONFIG.model_dict[args.train]} with dataset ag_news")
        train_model(args.train, get_dataset())
        
    elif args.predict:    
        from src.models.predict_model import predict_model
        from src.models.config import CONFIG
        print(f"↗️  Predicting by {CONFIG.model_dict[args.predict]} with test dataset ag_news")
        predict_model(args.predict, get_dataset())   
        
    else:
        print(usage_text)       
if __name__ == "__main__":
    main()
