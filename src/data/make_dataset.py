from datasets import load_dataset

def load_and_save_data():
    dataset = load_dataset("fancyzhx/ag_news")
    dataset.save_to_disk("data/raw/")
    return dataset
