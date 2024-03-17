from PyQt5 import QtWidgets, uic
import os
from lib.mylibrary import Mylibrary

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        ui_path = os.path.join(os.path.dirname(__file__), '..', 'ui', 'mainwindow.ui')
        uic.loadUi(ui_path, self)

        # Kết nối nút bấm (giả sử nút có tên là 'greetButton' và label là 'greetingLabel')
        self.greetButton.clicked.connect(self.show_greeting)

    def show_greeting(self):
        name = self.nameLineEdit.text()  # Giả sử QLineEdit có tên là 'nameLineEdit'
        greeting = Mylibrary.create_greeting(name)
        self.greetingLabel.setText(greeting)  # Giả sử QLabel có tên là 'greetingLabel'
