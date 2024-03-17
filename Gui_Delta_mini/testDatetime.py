import sys
from PyQt6.QtWidgets import QApplication, QWidget, QDateTimeEdit, QLabel, QFormLayout


class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle('PyQt QDateTimeEdit')
        self.setMinimumWidth(200)

        layout = QFormLayout()
        self.setLayout(layout)

        self.datetime_edit = QDateTimeEdit(self, calendarPopup=True)
        self.datetime_edit.dateTimeChanged.connect(self.update)

        self.result_label = QLabel('', self)

        layout.addRow('Date:', self.datetime_edit)
        layout.addRow(self.result_label)

        self.show()

    def update(self):
        value = self.datetime_edit.dateTime()
        self.result_label.setText(value.toString("yyyy-MM-dd HH:mm"))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())