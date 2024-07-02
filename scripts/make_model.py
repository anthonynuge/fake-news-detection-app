from sklearn.model_selection import train_test_split
import os
import sys
import pandas as pd


def make_model():
    # Cross support for windows and linux
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)

    from src.pipelines import create_best_gbc_pipe
    from utils.model_utils import save_model
    from src.model_training import train_report

    model_path = os.path.join(project_root, "models", "final_model.pk1")
    data_path = os.path.join(project_root, "data", "raw", "fake_and_real_news.csv")

    data = pd.read_csv(data_path)
    mapping = {"Fake": 1, "Real": 0}
    data["label_binary"] = data["label"].map(mapping)
    data.drop_duplicates(subset="Text", inplace=True)

    X = data["Text"]
    y = data["label_binary"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=2024
    )
    pipe = create_best_gbc_pipe()
    train_report(
        pipe, "Current Model (GBC_Mean_Vector)", X_train, y_train, X_test, y_test
    )
    save_model(pipe, model_path)


if __name__ == "__main__":
    make_model()
