def predict_model(model, dataset):
    from src.features.utils import load_trained_model
    from src.models.config import CONFIG
    from src.features.evaluation import evaluate_accuracy, plot_confusion_matrix, evaluate_classification, print_classification_report, evaluate_roc_auc, print_cross_validation
    from src.features.timer import TimerLogger

    timer = TimerLogger(task_type="Predicting") 
    timer.start()
    
    model_name = CONFIG.model_dict[model]
    print(f"🔮 Making predictions with {model_name}...")
    
    trained_model = load_trained_model(model)
    X_test = dataset["test"]["text"]
    y_test  = dataset["test"]["Category"]
    
    predictions = trained_model.predict(X_test)
    
    timer.stop()
    
    predictions = map_predictions(predictions)
    
    print("=== Test Set Performance ===")
    evaluate_accuracy(y_test , predictions)
    evaluate_classification(y_test, predictions)
    print_classification_report(y_test, predictions, labels=CONFIG.categories)

    if model not in ["neural_network", "hidden_markov_model", "bayesian_network", "svm", "svm_pca", "discriminative", "bagging", "boosting"]:
        print("=== Cross-Validation on Test Set ===")
        print_cross_validation(trained_model, dataset["train"]["text"], dataset["train"]["Category"])
        
    if model in ["bayesian_network"]:
        evaluate_roc_auc(y_test, predictions, trained_model)
    
    plot_confusion_matrix(y_test, predictions, labels=CONFIG.categories, model = model)
    return predictions

def naive_bayes(dataset):
    from src.features.utils import load_trained_model
    from src.models.config import CONFIG
    from src.features.evaluation import evaluate_accuracy, plot_confusion_matrix, evaluate_classification, print_classification_report, evaluate_roc_auc, print_cross_validation
    
    trained_model = load_trained_model("naive_bayes")
    X_test = dataset["test"]["text"]
    y_test  = dataset["test"]["Category"]
        
    predictions = trained_model.predict(X_test)
    
    print("=== Test Set Performance ===")
    evaluate_accuracy(y_test , predictions)
    evaluate_classification(y_test, predictions)
    print_classification_report(y_test, predictions, labels=CONFIG.categories)

    print("=== Cross-Validation on Training Set ===")
    print_cross_validation(trained_model, dataset["train"]["text"], dataset["train"]["Category"])
    
    plot_confusion_matrix(y_test, predictions, labels=CONFIG.categories)
    return predictions
    
def decision_tree(dataset):
    from src.features.utils import load_trained_model
    from src.models.config import CONFIG
    from src.features.evaluation import evaluate_accuracy, plot_confusion_matrix, evaluate_classification, print_classification_report, evaluate_roc_auc, print_cross_validation
    
    trained_model = load_trained_model("decision_tree")
    X_test = dataset["test"]["text"]
    y_test  = dataset["test"]["Category"]
        
    predictions = trained_model.predict(X_test)
    
    print("=== Test Set Performance ===")
    evaluate_accuracy(y_test , predictions)
    evaluate_classification(y_test, predictions)
    print_classification_report(y_test, predictions, labels=CONFIG.categories)

    print("=== Cross-Validation on Training Set ===")
    print_cross_validation(trained_model, dataset["train"]["text"], dataset["train"]["Category"])
    
    plot_confusion_matrix(y_test, predictions, labels=CONFIG.categories)
    return predictions

def neural_network(dataset):
    from src.features.utils import load_trained_model
    from src.models.config import CONFIG
    from src.features.evaluation import evaluate_accuracy, plot_confusion_matrix, evaluate_classification, print_classification_report, evaluate_roc_auc, print_cross_validation
    
    trained_model = load_trained_model("neural_network")
    X_test = dataset["test"]["text"]
    y_test  = dataset["test"]["Category"]
        
    predictions = trained_model.predict(X_test)
    
    print("=== Test Set Performance ===")
    evaluate_accuracy(y_test , predictions)
    evaluate_classification(y_test, predictions)
    print_classification_report(y_test, predictions, labels=CONFIG.categories)

    plot_confusion_matrix(y_test, predictions, labels=CONFIG.categories)
    return predictions

def bayesian_network(dataset):
    from src.features.utils import load_trained_model
    from src.models.config import CONFIG
    from src.features.evaluation import evaluate_accuracy, plot_confusion_matrix, evaluate_classification, print_classification_report, evaluate_roc_auc, print_cross_validation
    from src.features.timer import TimerLogger
    
    timer = TimerLogger(interval=10) 
    timer.start()
    
    trained_model = load_trained_model("bayesian_network")
    X_test = dataset["test"]["text"]
    y_test  = dataset["test"]["Category"]
        
    predictions = trained_model.predict(X_test)
    
    timer.stop()
    
    print("=== Test Set Performance ===")
    evaluate_accuracy(y_test , predictions)
    evaluate_classification(y_test, predictions)
    print_classification_report(y_test, predictions, labels=CONFIG.categories)
    evaluate_roc_auc(y_test, predictions, trained_model)
    
    plot_confusion_matrix(y_test, predictions, labels=trained_model.label_encoder.classes_)
    return predictions

def hidden_markov_model(dataset):
    from src.features.utils import load_trained_model
    from src.models.config import CONFIG
    from src.features.evaluation import evaluate_accuracy, plot_confusion_matrix, evaluate_classification, print_classification_report, evaluate_roc_auc, print_cross_validation
    
    trained_model = load_trained_model("hidden_markov_model")
    X_test = dataset["test"]["text"]
    y_test  = dataset["test"]["Category"]
        
    predictions = trained_model.predict(X_test)
    
    print("=== Test Set Performance ===")
    evaluate_accuracy(y_test , predictions)
    evaluate_classification(y_test, predictions)
    print_classification_report(y_test, predictions, labels=CONFIG.categories)
    
    plot_confusion_matrix(y_test, predictions, labels=CONFIG.categories)
    return predictions

def map_predictions(predictions):
    CATEGORY_MAPPING = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    value = [item[1] for item in list(CATEGORY_MAPPING.items())]
    return [CATEGORY_MAPPING[pred] for pred in predictions] if predictions[0] not in value else predictions
