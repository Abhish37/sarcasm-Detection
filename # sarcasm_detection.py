# sarcasm_detection_demo.py
# Concise, demo-friendly version: shows only confusion matrix and user input

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")

# STEP 1: LOAD AND PREPARE DATA
data = pd.read_json(r"d:\Sarcasm\Sarcasm_Headlines_Dataset.json", lines=True)
X = data["headline"]
y = data["is_sarcastic"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# STEP 2: TF-IDF VECTORIZATION (with n-grams)
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=8000,
    ngram_range=(1, 2)
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# STEP 3: MODEL TRAINING (with simple tuning)
param_grid = {"C": [1, 10], "class_weight": [None, "balanced"]}
grid = GridSearchCV(
    LogisticRegression(max_iter=500),
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)
grid.fit(X_train_tfidf, y_train)
model = grid.best_estimator_

# STEP 4: CONFUSION MATRIX ONLY
y_pred = model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Not Sarcastic", "Sarcastic"],
    yticklabels=["Not Sarcastic", "Sarcastic"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# STEP 5: INTERACTIVE USER INPUT
print("\nEnter sentences for sarcasm detection. Type 'done' when finished:")
test_sentences = []
while True:
    sentence = input("Sentence: ")
    if sentence.lower() == "done":
        break
    if sentence.strip():
        test_sentences.append(sentence.strip())

if not test_sentences:
    print("\nNo test sentences provided. Exiting.")
else:
    test_features = vectorizer.transform(test_sentences)
    probs = model.predict_proba(test_features)[:, 1]
    preds = model.predict(test_features)

    print("\nPredictions:\n")
    for s, p, prob in zip(test_sentences, preds, probs):
        label = "Sarcastic" if p == 1 else "Not Sarcastic"
        print(f"→ {s}\n   Prediction: {label} (Confidence: {prob:.2f})\n")


# Sample test case
# Sentence: Oh great, another meeting.
# Sentence: The sun rises in the east.
