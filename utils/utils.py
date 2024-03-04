from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def print_eval_metrics(y_true, y_pred):
    # Get eval metrics
    f1, accuracy, precision, recall = get_eval_metrics(y_true, y_pred)

    # Print metrics
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")


def get_eval_metrics(y_true, y_pred):
    """
    Calculate and print F1 score, accuracy, precision, and recall for the given true labels and predicted labels.

    Parameters:
    - y_true: array-like of shape (n_samples,) True labels.
    - y_pred: array-like of shape (n_samples,) Predicted labels.
    """
    # Calculate metrics
    f1 = f1_score(y_true, y_pred, average="weighted")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")

    return f1, accuracy, precision, recall
