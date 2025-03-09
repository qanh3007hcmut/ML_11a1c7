def train_model_classifiers(model, dataset):
    # Import libs
    from src.features.utils import tune_hyperparameters, build_pipeline, save_trained_model
    from src.features.timer import TimerLogger
    from src.models.config import CONFIG
    
    model_name = CONFIG.model_dict[model]
    
    timer = TimerLogger() 
    timer.start()
    
    # Extract dataset
    X_train = dataset["train"]["text"]
    y_train = dataset["train"]["Category"]
    
    # Build and train the model
    pipeline = build_pipeline(model)
    tuned_pipeline = tune_hyperparameters(pipeline, X_train, y_train)
    tuned_pipeline.fit(X_train, y_train)

    timer.stop()
    # Save the trained model
    save_trained_model(tuned_pipeline, model)

    print(f"✅ {model_name} is trained")
    return tuned_pipeline

def naive_bayes(dataset):
    
    # Import libs
    from src.features.utils import tune_hyperparameters, build_pipeline, save_trained_model
    from src.features.timer import TimerLogger
    
    timer = TimerLogger() 
    timer.start()
    
    # Extract dataset
    X_train = dataset["train"]["text"]
    y_train = dataset["train"]["Category"]
    
    # Build and train the model
    pipeline = build_pipeline("naive_bayes")
    tuned_pipeline = tune_hyperparameters(pipeline, X_train, y_train)
    tuned_pipeline.fit(X_train, y_train)

    timer.stop()
    # Save the trained model
    save_trained_model(tuned_pipeline, "naive_bayes")

    print(f"✅ Naive Bayes is trained")
    return tuned_pipeline

def decision_tree(dataset):
    
    # Import libs
    from src.features.utils import tune_hyperparameters, build_pipeline, save_trained_model
    from src.features.timer import TimerLogger
    
    timer = TimerLogger(interval=10) 
    timer.start()
    
    # Extract dataset
    X_train = dataset["train"]["text"]
    y_train = dataset["train"]["Category"]
    
    # Build and train the model
    pipeline = build_pipeline("decision_tree")
    tuned_pipeline = tune_hyperparameters(pipeline, X_train, y_train)
    tuned_pipeline.fit(X_train, y_train)

    timer.stop()
    # Save the trained model
    save_trained_model(tuned_pipeline, "decision_tree")

    print(f"✅ Decision Tree is trained")
    return tuned_pipeline

def neural_network(dataset):
    
    # Import libs
    from src.features.utils import build_pipeline, save_trained_model
    from src.features.timer import TimerLogger
    
    timer = TimerLogger(interval=10) 
    timer.start()
    
    # Extract dataset
    X_train = dataset["train"]["text"]
    y_train = dataset["train"]["Category"]
    
    # Build and train the model
    pipeline = build_pipeline("neural_network")
    pipeline.fit(X_train, y_train)

    timer.stop()
    # Save the trained model
    save_trained_model(pipeline, "neural_network")

    print(f"✅ Neural Network is trained")
    return pipeline

def bayesian_network(dataset):
    
    # Import libs
    from src.features.utils import build_pipeline, save_trained_model
    from src.features.timer import TimerLogger
    
    timer = TimerLogger(interval=10) 
    timer.start()
    
    # Extract dataset
    X_train = dataset["train"]["text"]
    y_train = dataset["train"]["Category"]
    
    # Build and train the model
    pipeline = build_pipeline("bayesian_network")
    pipeline.fit(X_train, y_train)

    timer.stop()
    # Save the trained model
    save_trained_model(pipeline, "bayesian_network")

    print(f"✅ Bayesian network is trained")
    return pipeline

def hidden_markov_model(dataset):
    
    # Import libs
    from src.features.utils import build_pipeline, save_trained_model
    from src.features.timer import TimerLogger
    
    timer = TimerLogger(interval=10) 
    timer.start()
    
    # Extract dataset
    X_train = dataset["train"]["text"]
    y_train = dataset["train"]["Category"]
    
    # Build and train the model
    pipeline = build_pipeline("hidden_markov_model")
    pipeline.fit(X_train, y_train)

    timer.stop()
    # Save the trained model
    save_trained_model(pipeline, "hidden_markov_model")

    print(f"✅ Hidden Markov Model is trained")
    return pipeline