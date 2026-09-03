# Fake News Detector

This is a small project I made that tries to guess if a news article is fake or real
just by looking at the text. It uses a dataset of about 6,300 news articles that are
already labeled FAKE or REAL, and trains a model on them using scikit-learn.

I originally just had a Jupyter notebook that trained the model and printed the
accuracy. I've since added a FastAPI backend and a Streamlit frontend so you can
actually type in an article and get a prediction, instead of just running it in a
notebook.

## How it works

1. `train.py` loads the CSV, cleans up the text a little (lowercase, removes links
   and punctuation), and turns it into numbers using TF-IDF.
2. It trains a PassiveAggressiveClassifier on that data (I tried Naive Bayes first
   since that's what I used originally, but this one got better accuracy when I
   tested it).
3. The trained model and vectorizer get saved to the `model/` folder using joblib.
4. `api/main.py` loads those saved files and exposes a `/predict` endpoint using
   FastAPI.
5. `app/streamlit_app.py` is just a simple form that sends whatever you type to the
   API and shows you the result.

## Accuracy

Got about **94% accuracy** on the test set with this version. The original notebook
(still included, `FakeNewsDetectorModel.ipynb`) got around 88% with plain Naive
Bayes and no text cleaning, so cleaning the text + switching the model + adding
bigrams to the vectorizer helped a decent amount.

```
              precision    recall  f1-score   support

        FAKE       0.96      0.92      0.94       604
        REAL       0.93      0.96      0.95       658

    accuracy                           0.94      1262
```

## How to run it

Install the requirements:

```
pip install -r requirements.txt
```

Train the model (this saves the model files into `model/`):

```
python train.py
```

Start the API:

```
uvicorn api.main:app --reload
```

Then in another terminal, start the frontend:

```
streamlit run app/streamlit_app.py
```

It should open in your browser. Type in a headline and some article text and hit
Check.

## Files

- `train.py` — trains and saves the model
- `api/main.py` — the FastAPI backend
- `app/streamlit_app.py` — the Streamlit frontend
- `data/news.csv` — the dataset
- `model/` — where the trained model gets saved (created after you run train.py)
- `FakeNewsDetectorModel.ipynb` — my original notebook version

## Things to note / limitations

- This is trained on a dataset that's mostly US politics news from around 2016, so
  it might not work that well on other topics or newer articles.
- It's really just picking up on writing style/word patterns, not actually checking
  facts. So a well written fake article could still get marked as real, and vice
  versa.
- This was a learning project so the code is pretty simple, there's no fancy error
  handling or anything like that in the API.

## Things I'd like to add later

- Better error handling in the API
- Maybe try a different/bigger dataset
- Deploy it somewhere so I don't have to run it locally
