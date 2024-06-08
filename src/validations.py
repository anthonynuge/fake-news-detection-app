import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np


def plot_roc_curves(models_list, X_train, X_test, y_train, y_test):
    plt.figure(figsize=(15, 6))
    print(models_list)

    for model_name, model in models_list:
        print(model_name)
        print(model)
        model.fit(X_train, y_train)
        y_probability = model.predict_proba(X_test)[:, 1]

        false_positive, true_positive, _ = roc_curve(y_test, y_probability)
        auc_score = roc_auc_score(y_test, y_probability)

        plt.plot(
            false_positive,
            true_positive,
            label=f"{model_name}, (AUC = {auc_score:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", label="Random Guesing")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()


# Cross Validation using Stratified splits
def cross_val_train_val_scores(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    train_scores = []
    val_scores = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)

        train_scores.append(train_score)
        val_scores.append(val_score)

    return train_scores, val_scores


def plot_train_val_scores(train_scores, val_scores, model_name):
    epochs = range(1, len(train_scores) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_scores, "bo-", label="Training Accuracy")
    plt.plot(epochs, val_scores, "ro-", label="Validation Accuracy")
    plt.title(f"Training and Validation Accuracy for {model_name}")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
