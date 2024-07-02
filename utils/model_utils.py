import pickle
import os


def save_model(model, path):
    with open(path, "wb") as destination:
        pickle.dump(model, destination)


def load_model(path):
    with open(path, "rb") as model:
        return pickle.load(model)


def model_exists(path):
    return os.path.exists(path)
