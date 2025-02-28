import argparse
from src.data.preprocess import load_and_preprocess_data
from src.models.train_model import naive_bayes as train_NB
from src.models.predict_model import naive_bayes as predict_NB
from src.features.evaluation import *
from src.features.ultils import get_dataset
def main():
    parser = argparse.ArgumentParser(description="Run various parts of the ML pipeline.")
    
    # Add pre action  
    parser.add_argument(
        "--task", 
        type=str, 
        choices=["preprocess","review"],
        help="Task to execute: fetch, save and review data"
    )
    
    # Add train action with choice of model
    parser.add_argument(
        "--train", 
        type=str, 
        choices=["naive_bayes"],
        help="choose model to train with dataset ag_news"
    )
    
    # Add predict action with choice of model
    parser.add_argument(
        "--predict", 
        type=str, 
        choices=["naive_bayes"],
        help="choose model to predict with dataset ag_news test"
    )
    
    # Add evaluation actio with choice of methods
    parser.add_argument(
        "--evaluate", 
        type=str, 
        choices=["accuracy", "confusion", "classification", "report", "roc"],
        help="choose method to evaluate trained model"
    )
    
    args = parser.parse_args()

    if args.task :
        if args.task == "preprocess":
            print("📥 Fetching and preprocess dataset...")
            load_and_preprocess_data()
        
        elif args.task == "review":
            print("Reviewing dataset...")
            get_dataset()
    elif args.train:    
        if args.train == "naive_bayes":
            print("↗️  Training Naive Bayes with dataset ag_news")
            train_NB(get_dataset())
    elif args.predict:    
        if args.predict == "naive_bayes":
            print("↗️  Predicting by Naive Bayes with test dataset ag_news")
            predict_NB(get_dataset())      
              
if __name__ == "__main__":
    main()
