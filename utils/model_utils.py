import pickle


def save_model(model, path):
    with open(path, "wb") as destination:
        pickle.dump(model, destination)


def load_model(path):
    with open(path, "rb") as model:
        return pickle.load(model)
