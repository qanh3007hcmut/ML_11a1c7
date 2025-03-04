from datasets import DatasetDict, Dataset
from src.data.make_dataset import load_and_save_data
from src.models.config import CONFIG

CATEGORY_MAPPING = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

def load_and_preprocess_data():
    dataset = load_and_save_data()
    
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    train_df = train_data.to_pandas()
    test_df = test_data.to_pandas()
        
    train_df['Category'] = train_df['label'].map(CATEGORY_MAPPING)
    test_df['Category'] = test_df['label'].map(CATEGORY_MAPPING)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    processed_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    processed_dataset.save_to_disk(CONFIG.processed_data_path)

    print(f"✅ Dữ liệu đã được xử lý và lưu tại: {CONFIG.processed_data_path}")

    return processed_dataset
