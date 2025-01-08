import pandas as pd
import re
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[{}]".format(re.escape(string.punctuation)), "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    # Load data
    fake_df = pd.read_csv("data/Fake.csv")
    real_df = pd.read_csv("data/True.csv")

    # Label
    fake_df["label"] = 1  # 1 = Fake
    real_df["label"] = 0  # 0 = Real

    # Combine
    df = pd.concat([fake_df, real_df], ignore_index=True)

    df["subject"] = df["subject"].fillna("")  # ensure no NaN
    df["date"]    = df["date"].fillna("")
    df["text"]    = df["text"].fillna("")

    # Combine them
    df["combined_text"] = df["subject"].astype(str) + " " + df["date"].astype(str) + " " + df["text"].astype(str)

    # Clean
    df["cleaned_text"] = df["combined_text"].apply(clean_text)

    X = df["cleaned_text"]
    y = df["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # LogisticRegression
    model = LogisticRegression(
        max_iter=300,
        solver="saga",
        random_state=42
    )
    model.fit(X_train_vec, y_train)

    # Simple evaluation
    acc = model.score(X_test_vec, y_test)
    print("Accuracy:", acc)

    # Save to model folder
    with open("model/news_model.pkl", "wb") as f_model:
        pickle.dump(model, f_model)

    with open("model/news_vectorizer.pkl", "wb") as f_vec:
        pickle.dump(vectorizer, f_vec)

    print("Model and vectorizer saved in model/.")

if __name__ == "__main__":
    main()
