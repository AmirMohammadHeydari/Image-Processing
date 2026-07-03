from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)


def evaluate_model(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        pos_label="Chelsea"
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label="Chelsea"
    )

    print("=" * 40)
    print("Confusion Matrix")
    print(cm)

    print(f"\nAccuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")