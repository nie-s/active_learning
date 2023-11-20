from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QComboBox, QDesktopWidget
from sklearn.svm import SVC

from ActiveLearning import ActiveLearning


class InputWidget(QWidget):
    def __init__(self, mainWindow):
        super().__init__()
        self.w = mainWindow

        self.layout = QtWidgets.QGridLayout()
        self.layout.setAlignment(Qt.AlignHCenter)
        self.layout.setSpacing(20)
        self.setLayout(self.layout)

        self.input_label = QLabel(self)
        self.input_label.setText(str('Select target label:'))
        self.input_label.setFixedSize(200, 35)
        self.layout.addWidget(self.input_label, 1, 1)

        self.choose = QComboBox()
        self.choose.setCurrentIndex(0)
        self.choose.setFixedSize(200, 35)

        # self.labels = ['beaches', 'bus', 'dinosaurs', 'elephants', 'flowers', 'foods', 'horses', 'monuments',
        #                'mountains_and_snow', 'people_and_villages_in_Africa']

        self.choose.addItems(self.w.labels)
        self.layout.addWidget(self.choose, 2, 1)

        self.button = QtWidgets.QPushButton(self)
        self.button.setText('Confirm')
        self.button.clicked.connect(self.confirm)
        self.button.setGeometry(210, 100, 80, 35)

        self.layout.addWidget(self.button, 3, 1)

    def confirm(self):
        index = self.choose.currentIndex()
        self.w.target_label = self.w.labels[index]
        # train_feature_list, test_feature_list = extractor.extract_feature(target_label)
        self.w.train_feature_list, self.w.test_feature_list, self.w.test_labels = \
            self.w.extractor.extract_feature_local(self.w.target_label)

        self.w.active_learning = ActiveLearning(self.w.n_category, self.w.n_train, self.w.train_list,
                                                self.w.train_feature_list, self.w.labels, self.w.target_label)

        self.w.clf = SVC(gamma='auto')

        cw = ChooseWidget(self.w)
        self.w.setCentralWidget(cw)


class ChooseWidget(QWidget):
    def __init__(self, mainWindow):
        super().__init__()
        self.w = mainWindow

        self.ui = uic.loadUi('choose_widget.ui', self)

        self.imgs = [self.ui.label_1, self.ui.label_2, self.ui.label_3, self.ui.label_4, self.ui.label_5,
                     self.ui.label_6, self.ui.label_7, self.ui.label_8, self.ui.label_9, self.ui.label_10,
                     self.ui.label_11, self.ui.label_12, self.ui.label_13, self.ui.label_14, self.ui.label_15,
                     self.ui.label_16, self.ui.label_17, self.ui.label_18, self.ui.label_19, self.ui.label_20]

        self.boxes = [self.ui.checkBox_1, self.ui.checkBox_2, self.ui.checkBox_3, self.ui.checkBox_4,
                      self.ui.checkBox_5, self.ui.checkBox_6, self.ui.checkBox_7, self.ui.checkBox_8,
                      self.ui.checkBox_9, self.ui.checkBox_10, self.ui.checkBox_11, self.ui.checkBox_12,
                      self.ui.checkBox_13, self.ui.checkBox_14, self.ui.checkBox_15, self.ui.checkBox_16,
                      self.ui.checkBox_17, self.ui.checkBox_18, self.ui.checkBox_19, self.ui.checkBox_20]

        self.selected_labels = []
        self.selected_list, _, filenames = self.w.active_learning.random_sample(20)
        self.set_img(filenames)

        self.w.resize(1300, 980)
        screen = QDesktopWidget().screenGeometry()
        size = self.w.geometry()
        self.w.move((screen.width() - size.width()) / 2,
                    (screen.height() - size.height()) / 2 - 45)

        self.ui.confirm_button.clicked.connect(self.confirm)
        self.ui.result_button.clicked.connect(self.get_result)

    def set_img(self, filenames):
        for i in range(len(self.imgs)):
            lbl = self.imgs[i]
            filename = 'dataset/training_set/' + filenames[i]

            pixmap = QPixmap(filename)  # 按指定路径找到图片
            lbl.setPixmap(pixmap)  # 在label上显示图片
            lbl.setScaledContents(True)  # 让图片自适应label大小

    def confirm(self):

        for i in range(len(self.boxes)):
            if self.boxes[i].isChecked():
                self.selected_labels.append(self.w.target_label)
            else:
                self.selected_labels.append('other')

        print(self.selected_labels)
        self.w.clf.fit(self.selected_list, self.selected_labels)

        for i in range(len(self.boxes)):
            self.boxes[i].setChecked(False)

        distance_metric, benchmark = self.w.active_learning.get_distance(self.w.clf)
        self.selected_list, _, filenames = self.w.active_learning.active_sample(20)
        print(len(filenames))
        self.set_img(filenames)

    def get_result(self):
        top_k = 20
        filenames = self.w.active_learning.get_result(self.w.clf, top_k)
        self.set_img(filenames)

        # self.w.active_learning.show_result(self.w.active_learning.selected, self.w.train_list)
        # print(result)
