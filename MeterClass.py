import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATASET_PATH = "data/poems_dataset.json"
TEST_SIZE = 0.20
RANDOM_STATE = 42


df = pd.read_json(DATASET_PATH)

df = (
    df[["text", "sea"]]
    .dropna()
    .explode("text")
    .rename(columns={"text": "verse", "sea": "meter"})
    .reset_index(drop=True)
)

df["verse"] = df["verse"].astype(str).str.strip()
df["meter"] = df["meter"].astype(str).str.strip()

df = df[df["verse"] != ""]


def remove_harakat(text: str) -> str:
    """Remove Arabic diacritics (harakat) from text."""
    return re.sub(r"[\u064B-\u0652]", "", text)


# Uncomment to remove Arabic diacritics
# df["verse"] = df["verse"].apply(remove_harakat)


X_train, X_test, y_train, y_test = train_test_split(
    df["verse"],
    df["meter"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["meter"],
)


model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
            ),
        ),
    ]
)



model.fit(X_train, y_train)


predictions = model.predict(X_test)

print("=" * 50)
print("Accuracy")
print("=" * 50)
print(f"{accuracy_score(y_test, predictions):.4f}")

print("\n" + "=" * 50)
print("Macro F1 Score")
print("=" * 50)
print(f"{f1_score(y_test, predictions, average='macro'):.4f}")

print("\n" + "=" * 50)
print("Classification Report")
print("=" * 50)
print(classification_report(y_test, predictions))