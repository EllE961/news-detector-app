import sys
from PyQt5.QtWidgets import QApplication
from news_classifier_app import NewsClassifierApp

def main():
    app = QApplication(sys.argv)
    window = NewsClassifierApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
