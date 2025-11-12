# sarcasm_detection_instant.py
# Loads a pre-trained sarcasm detection model and runs instantly.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")

# STEP 1: LOAD MODEL + VECTORIZER
print("\nLoading saved model and vectorizer...")
model = joblib.load("sarcasm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
print("✅ Model loaded successfully!")

# STEP 2: EVALUATE ON SAMPLE DATA (Confusion Matrix)
data = pd.read_json(r"d:\Sarcasm\Sarcasm_Headlines_Dataset.json", lines=True)
X = data["headline"]
y = data["is_sarcastic"]

# Use only a small test subset for speed
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_test_tfidf = vectorizer.transform(X_test)
y_pred = model.predict(X_test_tfidf)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Not Sarcastic", "Sarcastic"],
    yticklabels=["Not Sarcastic", "Sarcastic"]
)
plt.title("Confusion Matrix (Pre-trained Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# STEP 3: INTERACTIVE USER INPUT
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
