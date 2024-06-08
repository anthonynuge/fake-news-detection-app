from sklearn.metrics import classification_report, confusion_matrix


def train_model(pipe, X_train, y_train):
    pipe.fit(X_train, y_train)


def train_report(pipe, pipe_name, X_train, y_train, X_test, y_test):
    train_model(pipe, X_train, y_train)
    pred = pipe.predict(X_test)
    print(f"{pipe_name} Report: ")
    print(classification_report(y_test, pred, target_names=["Real", "Fake"]))
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, pred))
    print("=" * 30)


def train_report_all(pipe_dict, X_train, y_train, X_test, y_test):
    for name, pipe in pipe_dict.items():
        train_report(pipe, name, X_train, y_train, X_test, y_test)
