from datasets import load_dataset
import pandas as pd

def get_dataset() :
    dataset = load_dataset("fancyzhx/ag_news")
    return dataset

def export_raw_dataset():
    dataset = get_dataset()
    df = pd.DataFrame(dataset["train"])  
    df.to_csv("raw_data.csv", index=False)
    print("Saved dataset to data/raw")
