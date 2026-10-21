"""Evaluate the shipped model and rebuild the README figures.

Run from the repo root:  python scripts/make_figures.py   (about 1.5 minutes)
Loads model/news_model.pkl and model/news_vectorizer.pkl as they are committed, rebuilds the exact
test split from news_training.py, and scores the model on different versions of the same test articles.
"""
import copy
import os
import pickle
import re
import string

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "assets")
os.makedirs(OUT, exist_ok=True)

BG, PANEL = "#0d0c0b", "#131211"
INK, INK2, INK3 = "#ede7de", "#aaa299", "#7a736c"
GRID = "#24221f"
EMBER, TEMPER, PURPLE, GREEN = "#d95926", "#3987e5", "#9085e9", "#199e70"
SERIES = [EMBER, TEMPER, PURPLE, GREEN, "#e8b44c", "#c0509c"]
HEAT = LinearSegmentedColormap.from_list("heat", ["#141110", "#2a1410", "#4b170c", "#7a1f0a", "#ad320c",
                                                  "#d9520f", "#f07a1e", "#f9a63b", "#fccf6b", "#fff0c2"])
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.titlesize": 12.5, "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "axes.axisbelow": True, "font.family": ["Segoe UI", "DejaVu Sans"], "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK2,
    "figure.dpi": 110, "savefig.dpi": 200,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")


def caption(fig, text):
    fig.text(0.012, -0.015, text, color=INK3, fontsize=8.5, ha="left", va="top")


# same cleaning as news_training.py and news_classifier_app.py
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[{}]".format(re.escape(string.punctuation)), "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_wire(text):
    text = re.sub(r"^.{0,120}?\(Reuters\)\s*-\s*", "", text)
    return re.sub(r"(?i)reuters", "", text)


def main():
    with open(os.path.join(ROOT, "model", "news_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(ROOT, "model", "news_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    fake = pd.read_csv(os.path.join(ROOT, "data", "Fake.csv"))
    real = pd.read_csv(os.path.join(ROOT, "data", "True.csv"))
    fake["label"], real["label"] = 1, 0
    df = pd.concat([fake, real], ignore_index=True)
    for col in ["subject", "date", "text"]:
        df[col] = df[col].fillna("").astype(str)
    df["combined_text"] = df["subject"] + " " + df["date"] + " " + df["text"]
    _, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    y = test["label"].values
    print(f"test articles {len(test)}  real {int((y == 0).sum())}  fake {int((y == 1).sum())}")
    print("subjects  fake:", sorted(fake["subject"].unique()), " real:", sorted(real["subject"].unique()))

    # what the app sends to the model: metadata box + " " + article box, then clean_text
    inputs = {
        "subject + date + text\n(how it was trained)": test["combined_text"],
        "text only\n(metadata box left empty)": " " + test["text"],
        "text only,\nReuters tags removed": " " + test["text"].apply(strip_wire),
        "placeholder example\n'politics 2020-05-12' + text": "politics 2020-05-12 " + test["text"],
    }
    results = {}
    for name, raw in inputs.items():
        pred = model.predict(vectorizer.transform(raw.apply(clean_text)))
        cm = confusion_matrix(y, pred, labels=[0, 1])
        results[name] = cm
        print(f"{name.replace(chr(10), ' '):52s} acc {np.trace(cm) / cm.sum():.4f}  "
              f"real->fake {cm[0, 1]:4d}  fake->real {cm[1, 0]:4d}")

    slim = copy.copy(vectorizer)
    del slim.stop_words_  # only kept for inspection; transform() never reads it
    print(f"vectorizer.stop_words_ holds {len(vectorizer.stop_words_):,} pruned terms; "
          f"pickled without them: {len(pickle.dumps(slim)) / 1e6:.2f} MB")

    names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    order = np.argsort(coef)
    print("vocabulary", len(names), " top real:", ", ".join(names[order[:10]]))
    print("top fake:", ", ".join(names[order[-10:]][::-1]))

    # inputs figure: errors split by direction
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    labels = list(results)
    x = np.arange(len(labels))
    r2f = np.array([results[k][0, 1] for k in labels])
    f2r = np.array([results[k][1, 0] for k in labels])
    ax.bar(x, r2f, 0.6, color=EMBER, label="real article called fake")
    ax.bar(x, f2r, 0.6, bottom=r2f, color=TEMPER, label="fake article called real")
    for xi, a, b in zip(x, r2f, f2r):
        acc = 1 - (a + b) / len(y)
        ax.text(xi, a + b + 6, f"{a + b:,} errors · accuracy {acc:.3f}", ha="center", color=INK2, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, (r2f + f2r).max() * 1.18)
    ax.set_ylabel(f"misclassified test articles (of {len(y):,})")
    ax.set_title("Same model, same 8,980 test articles, different input")
    ax.legend(loc="upper left")
    caption(fig, "Stratified 80/20 split from news_training.py (random_state=42). The shipped pickles are "
                 "scored as committed; nothing is retrained.")
    save(fig, "inputs.png")

    # weights figure: subject and date tokens highlighted
    subject_tokens = {clean_text(s) for s in df["subject"].unique()}
    months = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
              "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
              "oct", "nov", "dec"]

    def is_meta(tok):
        parts = tok.split()
        return any(p in subject_tokens or p.isdigit() for p in parts) or tok in subject_tokens or \
            (len(parts) == 2 and (parts[0] in subject_tokens or parts[0] in months))

    n = 12
    idx = np.concatenate([order[:n], order[-n:]])
    vals = coef[idx]
    colors = [PURPLE if is_meta(names[i]) else (TEMPER if v < 0 else EMBER) for i, v in zip(idx, vals)]
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    yy = np.arange(len(idx))
    ax.barh(yy, vals, color=colors, height=0.72)
    ax.set_yticks(yy, names[idx])
    ax.axvline(0, color=INK3, lw=0.8)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("logistic regression weight (negative = real, positive = fake)")
    ax.set_title("Strongest weights in the shipped model")
    ax.legend(handles=[Patch(color=EMBER, label="article word, toward fake"),
                       Patch(color=TEMPER, label="article word, toward real"),
                       Patch(color=PURPLE, label="from the subject or date column")], loc="upper left")
    save(fig, "weights.png")


if __name__ == "__main__":
    main()
