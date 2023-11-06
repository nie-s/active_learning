import os
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow

from FeatureExtractor import FeatureExtractor
from Widgets import InputWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.extractor = FeatureExtractor()
        self.path = os.getcwd() + '/dataset/'
        self.train_list, self.test_list, self.labels = self.extractor.load_data(self.path)

        self.resize(500, 400)
        self.setWindowTitle('Active learning')

        self.cw = InputWidget(self)
        self.setCentralWidget(self.cw)

        self.n_train = 90  # size of one category in training set
        self.n_test = 10  # size of one category in test set
        self.n_category = 10

        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    app.exec_()
