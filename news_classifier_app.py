import sys
import re
import string
import pickle
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QTextEdit, QPushButton, QLabel,
    QVBoxLayout, QMessageBox
)
from PyQt5.QtCore import Qt

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[{}]".format(re.escape(string.punctuation)), "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class NewsClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("News Detector App (with Source/Time)")

        # Load model/vectorizer
        try:
            with open("model/news_model.pkl", "rb") as f_model:
                self.model = pickle.load(f_model)
            with open("model/news_vectorizer.pkl", "rb") as f_vec:
                self.vectorizer = pickle.load(f_vec)
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "Model or vectorizer file not found in model/ folder.")
            sys.exit(1)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        self.layout = QVBoxLayout()

        # Metadata box (source/time)
        self.meta_box = QTextEdit()
        self.meta_box.setPlaceholderText("Optional: Paste source/time here (e.g., 'politics 2020-05-12')")
        self.layout.addWidget(self.meta_box)

        # Article text box
        self.text_box = QTextEdit()
        self.text_box.setPlaceholderText("Paste article text here...")
        self.layout.addWidget(self.text_box)

        # Check button
        self.button = QPushButton("Check News")
        self.button.clicked.connect(self.check_news)
        self.layout.addWidget(self.button)

        # Result label
        self.result_label = QLabel("Result will appear here.")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label)

        self.central_widget.setLayout(self.layout)

    def check_news(self):
        metadata  = self.meta_box.toPlainText()
        article   = self.text_box.toPlainText()
        combined  = metadata + " " + article
        cleaned   = clean_text(combined)

        if not cleaned.strip():
            QMessageBox.warning(self, "Warning", "Please enter some text or metadata.")
            return

        X_vec = self.vectorizer.transform([cleaned])
        pred  = self.model.predict(X_vec)[0]
        label = "Fake" if pred == 1 else "Real"
        self.result_label.setText(f"Prediction: {label}")
