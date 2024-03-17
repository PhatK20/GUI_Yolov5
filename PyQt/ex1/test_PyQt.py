import sys
from PyQt5.QtWidgets import QApplication, QWidget

class SimpleApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initializeUI()

    def initializeUI(self):
        self.setGeometry(100, 100, 400, 300)  # x, y, width, height
        self.setWindowTitle('PyQt5 Simple App')
        self.show()

def main():
    app = QApplication(sys.argv)
    window = SimpleApp()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
