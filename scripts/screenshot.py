"""Render the real app window with two held-out articles and save docs/assets/hero.png.

Run from the repo root on Windows:  python scripts/screenshot.py
The windows are rendered off screen (Qt.WA_DontShowOnScreen) at 2x scale and captured with QWidget.grab().
"""
import os
import sys

os.environ.setdefault("QT_SCALE_FACTOR", "2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from news_classifier_app import NewsClassifierApp  # noqa: E402

# both articles are in the 20% test split used by news_training.py
ARTICLES = [("data/Fake.csv", 22930), ("data/True.csv", 661)]
OUT = os.path.join(ROOT, "docs", "assets")


def main():
    app = QApplication(sys.argv)
    shots = []
    for path, row in ARTICLES:
        text = pd.read_csv(path).loc[row, "text"]
        win = NewsClassifierApp()
        win.setAttribute(Qt.WA_DontShowOnScreen, True)
        win.resize(600, 560)
        win.text_box.setPlainText(text)
        win.show()
        app.processEvents()
        win.check_news()  # same slot the "Check News" button calls
        app.processEvents()
        tmp = os.path.join(OUT, f"_shot_{row}.png")
        win.grab().save(tmp)
        print(path, row, win.result_label.text())
        shots.append(Image.open(tmp).convert("RGB"))
        win.close()

    gap, edge = 48, 2
    w = sum(s.width for s in shots) + gap * (len(shots) - 1) + edge * 2 * len(shots)
    h = max(s.height for s in shots) + edge * 2
    canvas = Image.new("RGB", (w, h), (13, 12, 11))
    x = 0
    for s in shots:
        canvas.paste(Image.new("RGB", (s.width + 2 * edge, s.height + 2 * edge), (36, 34, 31)), (x, 0))
        canvas.paste(s, (x + edge, edge))
        x += s.width + 2 * edge + gap
    canvas = canvas.resize((1600, round(h * 1600 / w)), Image.LANCZOS)
    canvas.save(os.path.join(OUT, "hero.png"), optimize=True)
    for path, row in ARTICLES:
        os.remove(os.path.join(OUT, f"_shot_{row}.png"))
    print("wrote docs/assets/hero.png", canvas.size)


if __name__ == "__main__":
    main()
