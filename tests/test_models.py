def test_model_classification(model_type):
    from src.features.ultils import load_trained_model
    from tests.test_data import test_texts
    from src.features.timer import TimerLogger
    
    logger = TimerLogger(task_type="Testing")
    logger.start()
    
    trained_model = load_trained_model(model_type)    
    X_test = test_texts
    
    predictions = trained_model.predict(X_test)
    
    logger.stop()
    print("Predicted Categories:", predictions)