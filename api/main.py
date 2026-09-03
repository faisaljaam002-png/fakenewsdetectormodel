# main.py
# simple api that loads the saved model and gives predictions

import re
import string
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fake News Detector API")

# load the model and vectorizer we saved from train.py
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


# same cleaning function as train.py, needs to match or predictions
# will be way off since the model wasn't trained on messy text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# this defines what the request body should look like
class Article(BaseModel):
    title: str = ""
    text: str


@app.get("/")
def home():
    return {"message": "fake news detector api is running, go to /docs to try it"}


@app.post("/predict")
def predict(article: Article):
    combined_text = article.title + " " + article.text
    cleaned = clean_text(combined_text)

    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]

    if prediction == 1:
        label = "REAL"
    else:
        label = "FAKE"

    return {"label": label}
