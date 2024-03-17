import sys
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtCore import QThread, QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.uic import loadUi
import serial.tools.list_ports
import DeltaKinematic_v3 as Delta


class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the user interface from the .ui file
        loadUi("DeltaRobot_v1.ui", self)
        self.program_started = False
        self.connected = False
        self.selected_port = None
        self.ser = None
        self.send_data = None

        self.init_connect_FK()
        self.init_connect_IK()
        self.init_connect_control()
        self.init_connect_setup_com_port()

    def init_connect_FK(self):
        self.pushButton_FK.clicked.connect(self.calculate_FK)

    def init_connect_IK(self):
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

    def process_send_data(self, output_string):
        self.send_data = output_string
        self.ser.write(self.send_data.encode())
        self.lineEdit_sent_data.setText(output_string)
        length = len(output_string) * self.dataBits / 8
        self.lineEdit_sent_length.setText(str(length))

    def calculate_FK(self):
        if self.program_started:
            self.theta1 = float(self.lineEdit_FK_th1.text())
            self.theta2 = float(self.lineEdit_FK_th2.text())
            self.theta3 = float(self.lineEdit_FK_th3.text())
            positions = Delta.Forward_Kinematic(self.theta1, self.theta2, self.theta3)
            self.Px = positions[1]
            self.Py = positions[2]
            self.Pz = positions[3]
            self.lineEdit_FK_Px.setText(str(self.Px))
            self.lineEdit_FK_Py.setText(str(self.Py))
            self.lineEdit_FK_Pz.setText(str(self.Pz))

            self.process_send_data(f"F{self.Px}B{self.Py}C{self.Pz}")

    def calculate_IK(self):
        if self.program_started:
            self.Px = float(self.lineEdit_IK_Px.text())
            self.Py = float(self.lineEdit_IK_Py.text())
            self.Pz = float(self.lineEdit_IK_Pz.text())
            self.theta = float(self.lineEdit_IK_th.text())
            theta = Delta.Inverse_Kinematic(self.Px, self.Py, self.Pz, self.theta)
            self.theta1 = theta[1]
            self.theta2 = theta[2]
            self.theta3 = theta[3]
            self.lineEdit_IK_th1.setText(str(self.theta1))
            self.lineEdit_IK_th2.setText(str(self.theta2))
            self.lineEdit_IK_th3.setText(str(self.theta3))

            self.process_send_data(f"I{self.theta1}A{self.theta2}B{self.theta3}C")

    def run(self):
        self.program_started = True
        self.process_send_data(f"Run")
        self.receiveData_thread.start()
        print("Program started")

    def stop(self):
        self.program_started = False
        self.process_send_data(f"Stop")
        print("Program stopped")

    def reset(self):
        if self.program_started:
            self.lineEdits = [
                self.lineEdit_sent_data,
                self.lineEdit_sent_length,
                self.lineEdit_FK_th1,
                self.lineEdit_FK_th2,
                self.lineEdit_FK_th3,
                self.lineEdit_FK_Px,
                self.lineEdit_FK_Py,
                self.lineEdit_FK_Pz,
                self.lineEdit_IK_th1,
                self.lineEdit_IK_th2,
                self.lineEdit_IK_th3,
                self.lineEdit_IK_Px,
                self.lineEdit_IK_Py,
                self.lineEdit_IK_Pz,
            ]
        for lineEdit in self.lineEdits:
            lineEdit.clear()

        self.process_send_data(f"Reset")

    def setHome(self):
        if self.program_started:
            self.theta1 = 0
            self.theta2 = 0
            self.theta3 = 0
            self.lineEdit_FK_th1.setText(str(self.theta1))
            self.lineEdit_FK_th2.setText(str(self.theta2))
            self.lineEdit_FK_th3.setText(str(self.theta3))
            self.calculate_FK()

            self.process_send_data(f"Home: 1")

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
            stopBits = int(self.comboBox_stopBits.currentText())
            try:
                # Thực hiện kết nối đến cổng COM
                self.ser = serial.Serial(
                    self.selected_port, baudrate, self.dataBits, parity, stopBits
                )
                self.connected = True
                self.label_state.setText("ON")

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MyMainWindow()
    main_window.show()
    sys.exit(app.exec())
