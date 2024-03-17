import sys
from PyQt6.QtGui import QImage, QPixmap, QGuiApplication
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot, QDateTime
from PyQt6.QtWidgets import QApplication, QMainWindow, QDialog
from PyQt6.uic import loadUi
import serial.tools.list_ports
import DeltaKinematic_v4 as Delta
import numpy as np
import cv2


class Reading_line_data(QObject):
    data_updated = pyqtSignal(float, float, float, float, float, float)
    data_finished = pyqtSignal()

    def __init__(self):
        super().__init__()

    def run(self):
        flag = 0
        # Đọc dữ liệu từ tập tin và phát tín hiệu khi có dữ liệu mới
        for theta1, theta2, theta3, Px, Py, Pz in Delta.read_line_data():
            self.data_updated.emit(theta1, theta2, theta3, Px, Py, Pz)
            if flag == 0:
                QThread.msleep(2400)
                flag = 1
            QThread.msleep(550)  # Đặt độ trễ 200ms

        # Phát tín hiệu khi dữ liệu đã được hiển thị hết
        self.data_finished.emit()


class Reading_triangle_data(QObject):
    data_updated = pyqtSignal(float, float, float, float, float, float)
    data_finished = pyqtSignal()

    def __init__(self):
        super().__init__()

    def run(self):
        flag = 0
        # Đọc dữ liệu từ tập tin và phát tín hiệu khi có dữ liệu mới
        for theta1, theta2, theta3, Px, Py, Pz in Delta.read_triangle_data():
            self.data_updated.emit(theta1, theta2, theta3, Px, Py, Pz)
            if flag == 0:
                QThread.msleep(3000)
                flag = 1
            QThread.msleep(550)  # Đặt độ trễ 200ms

        # Phát tín hiệu khi dữ liệu đã được hiển thị hết
        self.data_finished.emit()


class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.connected = False
        self.selected_port = None
        self.ser = None
        self.program_started = False
        self.send_data = None
        self.string = None

        self.line_data_thread = None
        self.line_data_reader = None

        self.triangle_data_thread = None
        self.triangle_data_reader = None

        # Biến cờ để kiểm tra xem thread đã hoàn thành hay chưa
        self.line_data_finished = False
        self.triangle_data_finished = False

        # Load the user interface from the .ui file
        loadUi("Gui_Delta/DeltaRobot_v6.ui", self)
        self.setWindowTitle("Delta robot control interface")

        self.init_connect_kinematic()
        self.init_connect_control()
        self.init_connect_setup_com_port()
        self.init_connect_camera()
        self.init_connect_mode()
        self.init_connect_checkBoard()
        self.init_dateTime()

    def init_connect_kinematic(self):
        self.pushButton_FK.clicked.connect(self.calculate_FK)
        self.pushButton_IK.clicked.connect(self.calculate_IK)

    def init_connect_control(self):
        self.pushButton_run.clicked.connect(self.run)
        self.pushButton_stop.clicked.connect(self.stop)
        self.pushButton_reset.clicked.connect(self.reset)
        self.pushButton_setHome.clicked.connect(self.setHome)

    def init_connect_setup_com_port(self):
        self.populate_com_ports()
        self.pushButton_connect.clicked.connect(self.connect_com_port)
        self.pushButton_disconnect.clicked.connect(self.disconnect_com_port)

    def init_connect_camera(self):
        self.black_pixmap = None  # Thêm biến để lưu pixmap màu đen
        self.create_black_pixmap()  # Gọi hàm để tạo pixmap màu đen
        self.timer1 = QTimer(self)
        self.timer1.timeout.connect(self.update_frame)
        self.camera = cv2.VideoCapture(1)

        self.pushButton_on_camera.clicked.connect(self.on_camera)
        self.pushButton_off_camera.clicked.connect(self.off_camera)

    def init_connect_mode(self):
        self.pushButton_line.clicked.connect(self.draw_line)
        self.pushButton_triangle.clicked.connect(self.draw_triangle)

        self.line_data_reader = Reading_line_data()
        self.line_data_thread = QThread()

        self.triangle_data_reader = Reading_triangle_data()
        self.triangle_data_thread = QThread()

        # Kết nối sự kiện data_updated của Reading_line_data với hàm update_line_data
        self.line_data_reader.data_updated.connect(self.update_line_data)
        self.line_data_reader.data_finished.connect(self.on_line_data_finished)

        self.triangle_data_reader.data_updated.connect(self.update_triangle_data)
        self.triangle_data_reader.data_finished.connect(self.on_triangle_data_finished)

        # Chuyển Reading_line_data và QThread vào cùng một luồng
        self.line_data_reader.moveToThread(self.line_data_thread)
        self.line_data_thread.started.connect(self.line_data_reader.run)

        self.triangle_data_reader.moveToThread(self.triangle_data_thread)
        self.triangle_data_thread.started.connect(self.triangle_data_reader.run)

    def init_connect_checkBoard(self):
        self.pushButton_check_1.clicked.connect(self.check_1)
        self.pushButton_check_2.clicked.connect(self.check_2)
        self.pushButton_check_3.clicked.connect(self.check_3)
        self.pushButton_check_4.clicked.connect(self.check_4)
        self.pushButton_check_5.clicked.connect(self.check_5)
        self.pushButton_check_6.clicked.connect(self.check_6)
        self.pushButton_check_7.clicked.connect(self.check_7)
        self.pushButton_check_8.clicked.connect(self.check_8)
        self.pushButton_check_9.clicked.connect(self.check_9)
        self.pushButton_check_10.clicked.connect(self.check_10)
        self.pushButton_check_11.clicked.connect(self.check_11)
        self.pushButton_check_12.clicked.connect(self.check_12)
        self.pushButton_check_13.clicked.connect(self.check_13)
        self.pushButton_check_14.clicked.connect(self.check_14)
        self.pushButton_check_15.clicked.connect(self.check_15)
        self.pushButton_check_16.clicked.connect(self.check_16)
        self.pushButton_check_17.clicked.connect(self.check_17)
        self.pushButton_check_18.clicked.connect(self.check_18)
        self.pushButton_check_19.clicked.connect(self.check_19)
        self.pushButton_check_20.clicked.connect(self.check_20)
        self.pushButton_check_21.clicked.connect(self.check_21)
        self.pushButton_check_22.clicked.connect(self.check_22)
        self.pushButton_check_23.clicked.connect(self.check_23)
        self.pushButton_check_24.clicked.connect(self.check_24)
        self.pushButton_check_25.clicked.connect(self.check_25)
        self.pushButton_check_26.clicked.connect(self.check_26)
        self.pushButton_check_27.clicked.connect(self.check_27)
        self.pushButton_check_28.clicked.connect(self.check_28)
        self.pushButton_check_29.clicked.connect(self.check_29)
        self.pushButton_check_30.clicked.connect(self.check_30)
        self.pushButton_check_31.clicked.connect(self.check_31)
        self.pushButton_check_32.clicked.connect(self.check_32)
        self.pushButton_check_xy.clicked.connect(self.check_xy)

    def checkBoard(self, number):
        if self.program_started:
            if number < 9:
                self.Px = 30 * (4.5 - number)
                self.Py = -30 * 1.5 
            elif number < 17:
                self.Px = 30 * (12.5 - number) 
                self.Py = -30 * 0.5
            elif number < 25:
                self.Px = 30 * (20.5 - number)
                self.Py = 30 * 0.5
            else:
                self.Px = 30 * (28.5 - number)
                self.Py = 30 * 1.5
            self.Pz = -500

            self.lineEdit_Px.setText(str(self.Px))
            self.lineEdit_Py.setText(str(self.Py))
            self.lineEdit_Pz.setText(str(self.Pz))

            self.calculate_IK()

    def check_1(self):
        self.checkBoard(1)

    def check_2(self):
        self.checkBoard(2)

    def check_3(self):
        self.checkBoard(3)

    def check_4(self):
        self.checkBoard(4)

    def check_5(self):
        self.checkBoard(5)

    def check_6(self):
        self.checkBoard(6)

    def check_7(self):
        self.checkBoard(7)

    def check_8(self):
        self.checkBoard(8)

    def check_9(self):
        self.checkBoard(9)

    def check_10(self):
        self.checkBoard(10)

    def check_11(self):
        self.checkBoard(11)

    def check_12(self):
        self.checkBoard(12)

    def check_13(self):
        self.checkBoard(13)

    def check_14(self):
        self.checkBoard(14)

    def check_15(self):
        self.checkBoard(15)

    def check_16(self):
        self.checkBoard(16)

    def check_17(self):
        self.checkBoard(17)

    def check_18(self):
        self.checkBoard(18)

    def check_19(self):
        self.checkBoard(19)

    def check_20(self):
        self.checkBoard(20)

    def check_21(self):
        self.checkBoard(21)

    def check_22(self):
        self.checkBoard(22)

    def check_23(self):
        self.checkBoard(23)

    def check_24(self):
        self.checkBoard(24)

    def check_25(self):
        self.checkBoard(25)

    def check_26(self):
        self.checkBoard(26)

    def check_27(self):
        self.checkBoard(27)

    def check_28(self):
        self.checkBoard(28)

    def check_29(self):
        self.checkBoard(29)

    def check_30(self):
        self.checkBoard(30)

    def check_31(self):
        self.checkBoard(31)

    def check_32(self):
        self.checkBoard(32)

    def check_xy(self):
        if self.program_started:
            self.Px = float(self.lineEdit_check_x.text())
            self.Py = float(self.lineEdit_check_y.text())
            self.Pz = -500

            self.lineEdit_Px.setText(str(self.Px))
            self.lineEdit_Py.setText(str(self.Py))
            self.lineEdit_Pz.setText(str(self.Pz))

            self.calculate_IK()

    def init_dateTime(self):
        # Tạo một QTimer để cập nhật giá trị thời gian mỗi giây
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)  # Cập nhật mỗi giây

    def update_time(self):
        # Tùy chỉnh QDateTimeEdit để hiển thị thời gian thực
        current_datetime = QDateTime.currentDateTime()
        self.dateTimeEdit.setDateTime(current_datetime)

    def init_setup(self):
        self.send_data = None
        self.string = None
        self.program_started = False
        # Biến cờ để kiểm tra xem thread đã hoàn thành hay chưa
        self.line_data_finished = False

        self.theta1 = 0
        self.theta2 = 0
        self.theta3 = 0
        positions = Delta.Forward_Kinematic(self.theta1, self.theta2, self.theta3)
        self.Px = positions[1]
        self.Py = positions[2]
        self.Pz = positions[3]

        self.lineEdit_theta1.setText(str(self.theta1))
        self.lineEdit_theta2.setText(str(self.theta2))
        self.lineEdit_theta3.setText(str(self.theta3))
        self.lineEdit_Px.setText(str(self.Px))
        self.lineEdit_Py.setText(str(self.Py))
        self.lineEdit_Pz.setText(str(self.Pz))

        self.lineEdit_check_x.setText(str(self.Px))
        self.lineEdit_check_y.setText(str(self.Py))

    def create_black_pixmap(self):
        # Tạo một pixmap màu đen với kích thước 640x480
        black_image = np.zeros((480, 640, 3), dtype=np.uint8)
        black_image.fill(0)  # Điền màu đen vào ảnh
        black_image = cv2.cvtColor(black_image, cv2.COLOR_BGR2RGB)
        black_qimage = QImage(
            black_image.data,
            black_image.shape[1],
            black_image.shape[0],
            QImage.Format.Format_RGB888,
        )
        self.black_pixmap = QPixmap.fromImage(black_qimage)

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QImage(
                frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888
            )
            self.label_camera.setPixmap(QPixmap.fromImage(frame))

    def on_camera(self):
        if self.program_started:
            if not self.timer1.isActive():
                self.timer1.start(16)  # 16 ms là khoảng thời gian giữa các khung hình
                self.pushButton_on_camera.setEnabled(False)
                self.pushButton_off_camera.setEnabled(True)

    def off_camera(self):
        if self.program_started:
            if self.timer1.isActive():
                self.timer1.stop()
                self.pushButton_on_camera.setEnabled(True)
                self.pushButton_off_camera.setEnabled(False)
                self.label_camera.setPixmap(self.black_pixmap)

    def process_send_data(self, output_string):
        self.send_data = output_string
        self.ser.write(self.send_data.encode("utf-8"))
        self.lineEdit_sent_data.setText(output_string)
        length = len(output_string) * self.dataBits / 8
        self.lineEdit_sent_length.setText(str(length))

    def format_number(self, num):
        sign = "-" if num < 0 else "+"
        num_str = str(abs(num)).zfill(4)
        return f"{sign}{num_str}"

    def string_value(self, num1, num2, num3):
        for_num1 = self.format_number(int(num1 * 100))
        for_num2 = self.format_number(int(num2 * 100))
        for_num3 = self.format_number(int(num3 * 100))
        return f"{for_num1}a{for_num2}b{for_num3}c+0000d"

    def create_padded_string(self):
        if len(self.string) < 24:
            # Nếu chuỗi không đủ 24 ký tự, thêm dấu cách vào sau chuỗi
            self.string += " " * (24 - len(self.string))

    def calculate_FK(self):
        if self.program_started:
            input_theta1 = float(self.lineEdit_theta1.text())
            input_theta2 = float(self.lineEdit_theta2.text())
            input_theta3 = float(self.lineEdit_theta3.text())
            positions = Delta.Forward_Kinematic(self.theta1, self.theta2, self.theta3)
            if (
                all(
                    0 <= input_theta <= 60
                    for input_theta in (input_theta1, input_theta2, input_theta3)
                )
                and positions[3] >= -510
            ):
                self.theta1 = input_theta1
                self.theta2 = input_theta2
                self.theta3 = input_theta3

                positions = Delta.Forward_Kinematic(
                    self.theta1, self.theta2, self.theta3
                )
                self.Px = positions[1]
                self.Py = positions[2]
                self.Pz = positions[3]
                self.lineEdit_Px.setText(str(self.Px))
                self.lineEdit_Py.setText(str(self.Py))
                self.lineEdit_Pz.setText(str(self.Pz))

                self.lineEdit_check_x.setText(str(self.Px))
                self.lineEdit_check_y.setText(str(self.Py))

                self.process_send_data(
                    self.string_value(self.theta1, self.theta2, self.theta3)
                )

            else:
                warning_dialog.show()

    def calculate_IK(self):
        if self.program_started:
            input_Px = float(self.lineEdit_Px.text())
            input_Py = float(self.lineEdit_Py.text())
            input_Pz = float(self.lineEdit_Pz.text())
            theta = Delta.Inverse_Kinematic(input_Px, input_Py, input_Pz)
            if (
                all(
                    0 <= input_theta <= 60
                    for input_theta in (theta[1], theta[2], theta[3])
                )
                and input_Pz >= -510
            ):
                self.Px = input_Px
                self.Py = input_Py
                self.Pz = input_Pz

                self.theta1 = theta[1]
                self.theta2 = theta[2]
                self.theta3 = theta[3]

                self.lineEdit_theta1.setText(str(self.theta1))
                self.lineEdit_theta2.setText(str(self.theta2))
                self.lineEdit_theta3.setText(str(self.theta3))

                self.lineEdit_check_x.setText(str(self.Px))
                self.lineEdit_check_y.setText(str(self.Py))

                self.process_send_data(
                    self.string_value(self.theta1, self.theta2, self.theta3)
                )
            else:
                warning_dialog.show()

    def run(self):
        if self.connected:
            self.program_started = True
            # self.string = 'Run'
            # self.create_padded_string()
            # self.process_send_data(self.string)
            print("Program started")

    def stop(self):
        if self.program_started:
            self.program_started = False
            self.string = "Stop"
            self.create_padded_string()
            self.process_send_data(self.string)
            print("Program stopped")

    def reset(self):
        if self.program_started:
            self.lineEdits = [
                self.lineEdit_sent_data,
                self.lineEdit_sent_length,
                self.lineEdit_theta1,
                self.lineEdit_theta2,
                self.lineEdit_theta3,
                self.lineEdit_Px,
                self.lineEdit_Py,
                self.lineEdit_Pz,
                self.lineEdit_check_x,
                self.lineEdit_check_y
            ]
            for lineEdit in self.lineEdits:
                lineEdit.clear()

            self.init_setup()
            self.process_send_data(
                self.string_value(self.theta1, self.theta2, self.theta3)
            )
            # self.string = 'Reset'
            # self.create_padded_string()
            # self.process_send_data(self.string)

    def setHome(self):
        if self.program_started:
            self.theta1 = 0
            self.theta2 = 0
            self.theta3 = 0
            self.lineEdit_theta1.setText(str(self.theta1))
            self.lineEdit_theta2.setText(str(self.theta2))
            self.lineEdit_theta3.setText(str(self.theta3))
            self.calculate_FK()

            self.process_send_data(
                self.string_value(self.theta1, self.theta2, self.theta3)
            )

    def populate_com_ports(self):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.comboBox_comPort.addItem(port.device)

    def connect_com_port(self):
        if not self.connected:
            self.selected_port = self.comboBox_comPort.currentText()
            baudrate = int(self.comboBox_baudRate.currentText())
            self.dataBits = int(self.comboBox_dataBits.currentText())
            parity_string = self.comboBox_parity.currentText()
            if parity_string == "None":
                parity = serial.PARITY_NONE
            elif parity_string == "Even":
                parity = serial.PARITY_EVEN
            else:
                parity = serial.PARITY_ODD
            stopBits = float(self.comboBox_stopBits.currentText())
            try:
                # Thực hiện kết nối đến cổng COM
                self.ser = serial.Serial(
                    self.selected_port, baudrate, self.dataBits, parity, stopBits
                )
                self.connected = True
                self.label_state.setText("ON")
                self.init_setup()
                print(f"Connected to {self.selected_port} at {baudrate} baud")
            except Exception as e:
                print(f"Error connecting to {self.selected_port}: {str(e)}")
        else:
            print("Already connected to a COM port.")

    def disconnect_com_port(self):
        if self.connected:
            self.ser.close()
            self.connected = False
            self.label_state.setText("OFF")
            print(f"Disconnected from {self.selected_port}")
        else:
            print("Not connected to any COM port.")

    @pyqtSlot(float, float, float, float, float, float)
    def update_line_data(self, theta1, theta2, theta3, Px, Py, Pz):
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        # self.calculate_theta_ref()
        self.Px = Px
        self.Py = Py
        self.Pz = Pz
        self.lineEdit_theta1.setText(str(self.theta1))
        self.lineEdit_theta2.setText(str(self.theta2))
        self.lineEdit_theta3.setText(str(self.theta3))
        self.lineEdit_Px.setText(str(self.Px))
        self.lineEdit_Py.setText(str(self.Py))
        self.lineEdit_Pz.setText(str(self.Pz))
        self.lineEdit_check_x.setText(str(self.Px))
        self.lineEdit_check_y.setText(str(self.Py))
        self.process_send_data(self.string_value(self.theta1, self.theta2, self.theta3))

    @pyqtSlot(float, float, float, float, float, float)
    def update_triangle_data(self, theta1, theta2, theta3, Px, Py, Pz):
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        # self.calculate_theta_ref()
        self.Px = Px
        self.Py = Py
        self.Pz = Pz
        self.lineEdit_theta1.setText(str(self.theta1))
        self.lineEdit_theta2.setText(str(self.theta2))
        self.lineEdit_theta3.setText(str(self.theta3))
        self.lineEdit_Px.setText(str(self.Px))
        self.lineEdit_Py.setText(str(self.Py))
        self.lineEdit_Pz.setText(str(self.Pz))
        self.lineEdit_check_x.setText(str(self.Px))
        self.lineEdit_check_y.setText(str(self.Py))
        self.process_send_data(self.string_value(self.theta1, self.theta2, self.theta3))

    @pyqtSlot()
    def on_line_data_finished(self):
        # Luồng đã kết thúc do hết dữ liệu
        self.line_data_thread.quit()

    @pyqtSlot()
    def on_triangle_data_finished(self):
        # Luồng đã kết thúc do hết dữ liệu
        self.triangle_data_thread.quit()

    def draw_line(self):
        if self.program_started:
            if (
                self.line_data_thread is not None
                and not self.line_data_thread.isRunning()
            ):
                if self.line_data_finished:
                    # Nếu thread đã hoàn thành, tạo mới Reading_line_data và kết nối lại
                    self.line_data_reader = Reading_line_data()
                    self.line_data_reader.data_updated.connect(self.update_line_data)
                    self.line_data_reader.data_finished.connect(
                        self.on_line_data_finished
                    )
                    self.line_data_reader.moveToThread(self.line_data_thread)
                    self.line_data_thread.started.connect(self.line_data_reader.run)
                    self.line_data_finished = False  # Đặt lại biến cờ

            self.line_data_thread.start()

    def draw_triangle(self):
        if self.program_started:
            if (
                self.triangle_data_thread is not None
                and not self.triangle_data_thread.isRunning()
            ):
                if self.line_data_finished:
                    # Nếu thread đã hoàn thành, tạo mới Reading_line_data và kết nối lại
                    self.triangle_data_reader = Reading_line_data()
                    self.triangle_data_reader.data_updated.connect(
                        self.update_line_data
                    )
                    self.triangle_data_reader.data_finished.connect(
                        self.on_line_data_finished
                    )
                    self.triangle_data_reader.moveToThread(self.triangle_data_thread)
                    self.triangle_data_thread.started.connect(
                        self.triangle_data_reader.run
                    )
                    self.triangle_data_finished = False  # Đặt lại biến cờ

            self.triangle_data_thread.start()

    def show(self):
        super().show()
        # Get the rectangle representing the window
        qr = self.frameGeometry()
        # Get the center point of the screen
        cp = QGuiApplication.primaryScreen().availableGeometry().center()
        # Move the rectangle's center point to the screen's center point
        qr.moveCenter(cp)
        # Move the window to the rectangle's top left point
        self.move(qr.topLeft())


class MessageDialog(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("Gui_Delta/warning_v1.ui", self)
        self.setWindowTitle("Warning Dialog")
        self.pushButton_ok.clicked.connect(self.close)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MyMainWindow()
    warning_dialog = MessageDialog()
    main_window.show()
    sys.exit(app.exec())
