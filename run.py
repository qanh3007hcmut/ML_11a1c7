import argparse
from src.data.preprocess import load_and_preprocess_data
from src.features.utils import review_data, get_dataset, gen_requirement
from src.features.evaluation import *
import warnings
warnings.simplefilter("ignore", category=UserWarning)

def main():
    from src.models.config import CONFIG
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
    
    3. Test with test_data:
       python run.py --task test <model_name>
       
    4. Train a Model:
       python run.py --train <model_name>
       
       Available models:
       - naive_bayes
       - decision_tree
       - neural_network
       - bayesian_network
       - hidden_markov_model
       
       Example:
       python run.py --train naive_bayes
       
    5. Predict Using a Model:
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
        choices=list(CONFIG.model_dict.keys()),
        help="choose model to train with dataset ag_news"
    )
    
    # Add predict action with choice of model
    parser.add_argument(
        "--predict", 
        type=str, 
        choices=list(CONFIG.model_dict.keys()),
        help="choose model to predict with dataset ag_news test"
    )
    
    # Add test action with choice of model
    parser.add_argument(
        "--test", 
        type=str, 
        choices=list(CONFIG.model_dict.keys()),
        help="choose model to test with custom test"
    )
    
    args = parser.parse_args()
    
    if not args: print(usage_text)
    else: 
        if args.task:
            if not args.task[0]:
                parser.error("No task specified. Use --task preprocess, review, or test <model_name>")
                
            elif args.task[0] == "preprocess":
                print("📥 Fetching and preprocess dataset...")
                load_and_preprocess_data()
            
            elif args.task[0] == "review":
                print("Reviewing dataset...")
                review_data()
            elif args.task[0] == "requirement":
                gen_requirement()
        
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
        
        elif args.test:
            from tests.test_models import test_model_classification
            
            model_name = args.test
            valid_models = list(CONFIG.model_dict.keys())
            if model_name not in valid_models:
                parser.error(f"Invalid model name. Choose from: {valid_models}")
            
            print("🛠 Testing the model...")
            test_model_classification(model_name)
                     
        else:
            print(usage_text)       
if __name__ == "__main__":
    main()
