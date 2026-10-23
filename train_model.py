
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = pd.read_csv("dataset/emails.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data["email_text"],
    data["label"],
    test_size=0.25,
    random_state=42,
    stratify=data["label"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(
    y_test, predictions, zero_division=0
))

joblib.dump(model, "email_model.pkl")

print("Model trained and saved successfully!")
