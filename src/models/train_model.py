def naive_bayes(dataset):
    
    # Import libs
    from sklearn.naive_bayes import MultinomialNB
    from src.features.ultils import convert_to_sparse, save_trained_model, get_best_alpha
    from src.models.config import STOP_WORDS, TFIDF_MAX_FEATURES
    
    # Extract dataset
    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    
    # Sparse feature
    X_train = convert_to_sparse(train_texts, STOP_WORDS, TFIDF_MAX_FEATURES)

    # Use GridSearchCV to find best alpha
    best_alpha = get_best_alpha(X_train,train_labels)
    
    # Initialize model & train
    model = MultinomialNB(alpha=best_alpha)
    model.fit(X_train, train_labels)

    save_trained_model(model,"naive_bayes")

    print(f"✅ Naive Bayes is trained")
    return model
