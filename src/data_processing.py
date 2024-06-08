import gensim.downloader as api
import spacy
import pandas as pd
import numpy as np

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin

# Use spacy for natural language processing tasks such as removing stopwords, punctions, and lemmatiztion
nlp = spacy.load("en_core_web_sm")
# python -m spacy download en_core_web_sm


# Non punction and stopwords are lemmatized return list of processed tokens
def process_article(article):
    spacy_doc = nlp(article)
    cleaned = []
    for text in spacy_doc:
        # Ignore punctions and stopwords
        if text.is_punct or text.is_stop:
            continue
        cleaned.append(text.lemma_)
    return cleaned


# Rejoin cleaned text tokens from processed_article back into article format
def get_cleaned_article(article):
    article = article.lower()
    cleaned = process_article(article)
    return " ".join(word for word in cleaned)


# Uses nltk's vader to calculate sentiment
class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    # Calculate negative score, neutral score, positve score, compound score of a list of text.
    # Saves values into an a list of list
    def get_sentiment_scores(self, corpus):
        sentiment_scores = []
        for article in corpus:
            sentiment = self.sia.polarity_scores(article)
            sentiment_scores.append(list(sentiment.values()))
        return sentiment_scores

    # Uses scores and returns a dataframe to be used for feature extraction
    def extract_sentiments_features(self, corpus):
        sentiment_scores = self.get_sentiment_scores(corpus)
        sentiment_frame = pd.DataFrame(
            sentiment_scores,
            columns=["neg_score", "neu_score", "pos_score", "compound_score"],
        )
        return sentiment_frame


# Calculates the mean vector using googles pretrained model for more accurate word embedding.
# Used to vectorize text in pipeline
class WordEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.w2v_model = api.load("word2vec-google-news-300")

    # Calculate mean vector of a single aritcle returns a vector length 300
    # Text should be cleaned (remove stopwords, punctutijons, numbers and lemmatized) for more accurate calculations and faster processing

    def get_mean_vector(self, article):
        cleaned_article = process_article(article)
        return self.w2v_model.get_mean_vector(cleaned_article)

    # Not used for training just for calucations so no fitting inolved hence return self
    def fit(self, X, y=None):
        # print ("Fit is being called")
        return self

    # Takes multiple aritcles and return mean vector for each used in pipelines.
    def transform(self, corpus):
        mean_vectors = [self.get_mean_vector(article) for article in corpus]
        return np.stack(mean_vectors)


if __name__ == "__main__":
    corpus = [
        "This is an example sentence for feature extraction.",
        "Hi my name is anthony.",
        "I hate burgers and cats, jk I do like cats .",
    ]

    print("Cleaning Test: ")
    for article in corpus:
        print(get_cleaned_article(article))

    print("Sentiment Analyzer Test: ")
    sa = SentimentAnalyzer()
    print(sa.extract_sentiments_features(corpus))

    print("WordEmdedder Test: ")
    we = WordEmbedder()
    for article in corpus:
        mean_vector = we.get_mean_vector(article)
        print(mean_vector)
