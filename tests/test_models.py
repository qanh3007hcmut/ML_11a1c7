def test_model_classification(model_type):
    from src.features.utils import load_trained_model
    from tests.test_data import test_texts
    from src.features.timer import TimerLogger

    logger = TimerLogger(task_type="Testing")
    logger.start()
    
    trained_model = load_trained_model(model_type)    
    X_test = test_texts

    predictions = trained_model.predict(X_test)
    
    logger.stop()
    
    print("Predicted Categories:", map_predictions(predictions))

def map_predictions(predictions):
    CATEGORY_MAPPING = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    value = [item[1] for item in list(CATEGORY_MAPPING.items())]
    return [CATEGORY_MAPPING[pred] for pred in predictions] if predictions[0] not in value else predictions

        