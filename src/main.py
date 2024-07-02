import sys
import os
from model_training import train_model
from gui import App
from data_processing import SentimentAnalyzer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.model_utils import model_exists, load_model


def main():
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    model_path = os.path.join(model_dir, "final_model.pk1")

    if model_exists(model_path):
        print(f"Model already present at {model_path}. Loading model ...")
        model = load_model(model_path)

    else:
        print(
            f"Model is not present at {model_path}. Generating new model this could take a couple of minutes ..."
        )
        from scripts.make_model import make_model

        make_model()

        if model_exists(model_path):
            model = load_model(model_path)
            print("Model has been trained and loaded.")
        else:
            print(
                "Error training and loading the model. Make sure necessary libraries are installed. Exiting application"
            )
            return

    sentiment_analyzer = SentimentAnalyzer()
    app = App(model, sentiment_analyzer)


if __name__ == "__main__":
    main()
