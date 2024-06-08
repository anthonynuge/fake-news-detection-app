import sys
import os
import pandas as pd

# Cross support for windows and linux
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processsing import get_cleaned_article
from src.data_processsing import SentimentAnalyzer

data_path = os.path.join(project_root, "data", "raw", "fake_and_real_news.csv")
processed_path = os.path.join(project_root, "data", "processed", "processed_data.csv")

data = pd.read_csv(data_path)
articles = data["Text"]

# Convert labels to binary
data["label_binary"] = data["label"].replace({"Fake": 1, "Real": 0})

# Add length to processed frame
data["length"] = data["Text"].apply(len)

# Add sentiment score columns to frame
sent_analyzer = SentimentAnalyzer()
sentiment_frame = sent_analyzer.extract_sentiments_features(articles)
data = data.join(sentiment_frame)

# Add cleaned text to processed frame
data["cleaned_text"] = data["Text"].apply(get_cleaned_article)

# Save processed dataframe
data.to_csv(processed_path, index=False)
