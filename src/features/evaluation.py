def evaluate_accuracy(y_true, y_pred):
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_true, y_pred)
    print(f"📊 Accuracy: {accuracy:.4f}")
    return accuracy

def plot_confusion_matrix(y_true, y_pred, labels):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def evaluate_classification(y_true, y_pred):
    from sklearn.metrics import precision_score, recall_score, f1_score

    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"🎯 Precision: {precision:.4f}")
    print(f"📥 Recall: {recall:.4f}")
    print(f"⚖️ F1-score: {f1:.4f}")

    return precision, recall, f1

def print_classification_report(y_true, y_pred, labels):
    from sklearn.metrics import classification_report

    report = classification_report(y_true, y_pred, target_names=labels)
    print("Classification Report:\n", report)

def print_cross_validation(pipeline, X_train, y_train):
    import numpy as np
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"CV Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}") 
    
def evaluate_roc_auc(y_true, y_pred, model):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    import numpy as np 
    
    # Convert y_true and y_pred from labels to numerical values by label_encoder
    label_encoder = model.label_encoder
    y_true_encoded = label_encoder.transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)

    # One-hot encoding for multiclasses
    classes = np.unique(y_true_encoded)
    y_true_binarized = label_binarize(y_true_encoded, classes=classes)
    y_pred_binarized = label_binarize(y_pred_encoded, classes=classes)

    # Calculate macro-average ROC AUC
    roc_auc = roc_auc_score(y_true_binarized, y_pred_binarized, average="macro", multi_class="ovr")
    
    print(f"📈 Macro-average ROC AUC: {roc_auc:.4f}")    

