from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from src.data_processing import WordEmbedder

from sklearn.decomposition import PCA

# Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


# Create all the pipelines and store into a array. Array can be used for batch processsing
# def create_all_pipelines():
#     nb_pipe = create_nb_pipe()
#     dtc_pipe = create_dtc_pipe()
#     lr_pipe = create_lr_pipe()
#     gbc_pipe = create_gbc_pipe()
#     return [nb_pipe, dtc_pipe, lr_pipe, gbc_pipe]
def create_all_pipelines():
    potential_pipes = {
        "nb_tfidf": create_nb_pipe(),
        "dt_tfidf": create_dtc_pipe(),
        "lr_tfidf": create_lr_pipe(),
        "gbc_mean_vec": create_gbc_pipe(),
        "gbc_mean_vec_pca": create_gbc_pca_pipe(),
    }

    return potential_pipes


# Multinomial Native Bayes Pipeline
# Text preprocesssing (stopwords, lemmatization) not done here
def create_nb_pipe():
    pipe = Pipeline(
        [
            (
                "vectorizer",
                CountVectorizer(
                    ngram_range=(1, 2),
                    max_features=3000,
                ),
            ),
            ("tfidf", TfidfTransformer()),
            ("classifier", MultinomialNB()),
        ]
    )
    return pipe


# Decision Tree  Pipeline
def create_dtc_pipe():
    pipe = Pipeline(
        [
            (
                "vectorizer",
                CountVectorizer(
                    ngram_range=(1, 2),
                    max_features=3000,
                ),
            ),
            ("tfidf", TfidfTransformer()),
            ("classifier", DecisionTreeClassifier()),
        ]
    )
    return pipe


# Logistice Regression Pipeline
def create_lr_pipe():
    pipe = Pipeline(
        [
            (
                "vectorizer",
                CountVectorizer(
                    ngram_range=(1, 2),
                    max_features=3000,
                ),
            ),
            ("tfidf", TfidfTransformer()),
            ("classifier", LogisticRegression()),
        ]
    )
    return pipe


# Gradient Boosting Classifier
def create_gbc_pipe():
    pipe = Pipeline(
        [
            ("mean_vector", WordEmbedder()),
            ("scalar", StandardScaler()),
            ("classifier", GradientBoostingClassifier()),
        ]
    )
    return pipe


def create_gbc_pca_pipe():
    pipe = Pipeline(
        [
            ("mean_vector", WordEmbedder()),
            ("scalar", StandardScaler()),
            ("pca", PCA(n_components=3)),
            ("classifier", GradientBoostingClassifier()),
        ]
    )
    return pipe
