# train.py
# trains the fake news model and saves it so the api can use it later

import pandas as pd
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report

# load the data
df = pd.read_csv("data/news.csv")

# first column is just an index, don't need it
df = df.iloc[:, 1:]
df.columns = ["title", "text", "label"]

# drop empty rows and duplicates
df = df.dropna()
df = df.drop_duplicates()

# combine title and text into one column
df["content"] = df["title"] + " " + df["text"]


# clean up the text a bit before feeding it to the vectorizer
# removes links, punctuation and numbers, lowercases everything
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["content"] = df["content"].apply(clean_text)

# turn labels into numbers, 0 = fake, 1 = real
df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})
df = df.dropna()

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42
)

# vectorize the text
# added ngram_range so it picks up 2-word phrases too, not just single words
# this alone bumped accuracy up a good bit from the first version
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# I originally used Naive Bayes but PassiveAggressiveClassifier worked better
# on this dataset when I tried it out, so switched to that
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# check how it did
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

# save the model and vectorizer so the api can load them later
# without saving the vectorizer too, the api wouldn't know how to
# turn new text into the same kind of numbers the model was trained on
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("saved model to model/model.pkl")

# quick sanity check with a made up headline
sample = ["Scientists confirm water is wet"]
sample_clean = [clean_text(s) for s in sample]
sample_vec = vectorizer.transform(sample_clean)
print("Test prediction:", model.predict(sample_vec), "(0 = fake, 1 = real)")
