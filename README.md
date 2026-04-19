# fakenewsdetectormodel
# 📰 Fake News Detector Model

A Machine Learning project that detects whether a news article is **real or fake** using Natural Language Processing (NLP).

This project is part of my **AI/ML learning journey 🚀**

---

## 🚀 Repository Name
**fakenewsdetectormodel**

---

## 📌 Project Overview

Fake news spreads quickly online and can mislead people.  
This project builds a simple yet effective model to classify news articles using machine learning.

We use:
- **TF-IDF Vectorizer** for converting text into numerical form
- **Naive Bayes Classifier** for prediction

---

## 📊 Dataset

The dataset (`news.csv`) contains:
- `title` → News headline  
- `text` → Full article text  
- `label` → Fake or Real news label  

---

## ⚙️ Technologies Used

- Python 🐍  
- Pandas  
- Scikit-learn  
- NLP (Natural Language Processing)  
- TF-IDF Vectorizer  
- Naive Bayes Classifier  

---

## 🧠 Model Workflow

1. Load dataset  
2. Clean missing values and duplicates  
3. Combine `title + text` into a single feature  
4. Convert text using **TF-IDF Vectorizer**  
5. Split data into training and testing sets  
6. Train **Naive Bayes model**  
7. Evaluate model performance  
8. Test with custom input  

---

## 📈 Model Performance

Evaluated using:
- Accuracy Score  
- Precision, Recall, F1-score  

---

## 🧪 Example Prediction

```python
sample = ["Breaking: Scientists confirm water is wet"]
