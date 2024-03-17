import sys
from PyQt5 import QtWidgets, uic

class YesNoApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(YesNoApp, self).__init__()
        uic.loadUi('PyQt/yesNo.ui', self)  # Load file .ui

        # Tìm các widget theo tên và kết nối sự kiện
        self.pushButton_yes = self.findChild(QtWidgets.QPushButton, 'pushButton_yes')
        self.pushButton_no = self.findChild(QtWidgets.QPushButton, 'pushButton_no')
        self.lineEdit = self.findChild(QtWidgets.QLineEdit, 'lineEdit')

        # Kết nối sự kiện clicked của các nút với các hàm xử lý
        self.pushButton_yes.clicked.connect(self.yes_clicked)
        self.pushButton_no.clicked.connect(self.no_clicked)

    def yes_clicked(self):
        self.lineEdit.setText("Yes")

    def no_clicked(self):
        self.lineEdit.setText("No")

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = YesNoApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
