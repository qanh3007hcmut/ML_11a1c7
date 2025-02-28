def naive_bayes(dataset):
    from src.features.ultils import load_trained_model, convert_to_sparse
    from src.models.config import STOP_WORDS, TFIDF_MAX_FEATURES, CATEGORY
    from src.features.evaluation import evaluate_accuracy, plot_confusion_matrix, evaluate_classification, print_classification_report, evaluate_roc_auc
    trained_model = load_trained_model("naive_bayes")
    test_texts = dataset["test"]["text"]
    test_labels  = dataset["test"]["label"]
    
    X_test = convert_to_sparse(test_texts,STOP_WORDS,TFIDF_MAX_FEATURES)
    
    predictions = trained_model.predict(X_test)
    
    evaluate_accuracy(test_labels , predictions)
    plot_confusion_matrix(test_labels, predictions, labels=CATEGORY)
    evaluate_classification(test_labels, predictions)
    print_classification_report(test_labels, predictions, labels=CATEGORY)
    evaluate_roc_auc(test_labels, trained_model.predict_proba(X_test), num_classes=4)

    return predictions
    