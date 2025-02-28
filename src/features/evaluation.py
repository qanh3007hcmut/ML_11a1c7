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
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
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
    print(report)


def evaluate_roc_auc(y_true, y_scores, num_classes=4):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    auc_score = roc_auc_score(y_true_bin, y_scores, average="weighted", multi_class="ovr")
    
    print(f"📈 ROC-AUC Score: {auc_score:.4f}")
    return auc_score

