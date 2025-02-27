from make_dataset import get_dataset

CATEGORY_MAPPING = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

def load_and_preprocess_data():
    dataset = get_dataset()
    
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    train_df = train_data.to_pandas()
    test_df = test_data.to_pandas()
        
    train_df['Category'] = train_df['label'].map(CATEGORY_MAPPING)
    test_df['Category'] = test_df['label'].map(CATEGORY_MAPPING)
        
    return train_df, test_df