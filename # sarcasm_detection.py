# sarcasm_detection.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load Dataset
# Download the dataset from: https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection
data = pd.read_json(r"d:\Sarcasm\Sarcasm_Headlines_Dataset.json", lines=True)
X = data['headline']
y = data['is_sarcastic']

# Step 2: Split into Train/Test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 4: Model Training using Logistic Regression
model = LogisticRegression(max_iter=300)
model.fit(X_train_tfidf, y_train)

# Step 5: Prediction
y_pred = model.predict(X_test_tfidf)

# Step 6: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Test on Sample Inputs
test_sentences = []

print("\nEnter sentences for sarcasm detection. Type 'done' when finished:")

while True:
    sentence = input("Sentence: ")
    if sentence.lower() == 'done':
        break
    if sentence.strip():  # ignore empty input
        test_sentences.append(sentence.strip())

if not test_sentences:
    print("\nNo test sentences provided. Exiting.")
else:
    print("\nTest Sentences:")
    for s in test_sentences:
        print("-", s)

    # Vectorize test sentences
    test_features = vectorizer.transform(test_sentences)
    predictions = model.predict(test_features)

    print("\nPredictions:")
    for sentence, pred in zip(test_sentences, predictions):
        label = "Sarcastic" if pred == 1 else "Not Sarcastic"
        print(f"→ {sentence} → {label}")
