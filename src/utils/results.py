from logging import Logger

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def get_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    return (tp + tn) / (tp + tn + fp + fn)


def get_recall(tp: int, fn: int) -> float:
    return tp / (tp + fn)


def get_false_positive_rate(fp: int, tn: int) -> float:
    return fp / (fp + tn)


def get_false_negative_rate(fn: int, tp: int) -> float:
    return fn / (fn + tp)


def log_results(y_true, y_pred, logger: Logger):
    # Print classification report
    target_names = ['Human', 'AI']
    logger.info(classification_report(y_true,
                                      y_pred,
                                      target_names=target_names))

    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=['Actual Human', 'Actual AI'],
        columns=['Predicted Human', 'Predicted AI']
    )
    logger.info(cm_df.to_string())

    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = get_accuracy(tp, tn, fp, fn)
    recall = get_recall(tp, fn)
    false_positive_rate = get_false_positive_rate(fp, tn)
    false_negative_rate = get_false_negative_rate(fn, tp)

    # Print metrics
    logger.info("")
    logger.info(f"Accuracy: {accuracy * 100:.2f}%")
    logger.info(f"Recall: {recall * 100:.2f}%")
    logger.info(f"False Positive Rate: {false_positive_rate * 100:.2f}%")
    logger.info(f"False Negative Rate: {false_negative_rate * 100:.2f}%")
    logger.info("")
