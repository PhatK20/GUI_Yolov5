import os
from math import degrees, radians, atan2, sqrt, pi, sin, cos
import threading
import time

f = 110  # the side of the fixed triangle
e = 60  # the side of the end effector triangle
rf = 157  # the length of the upper joint
re = 440  # the length of the parallelogram joint
dz = -20


def check_solution(x, y, z, x1, y1, z1, x2, y2, z2, x3, y3, z3):
    check = 0
    if z <= 0:
        Eq1 = (x - x1) * (x - x1) + (y - y1) * (y - y1) + (z - z1) * (z - z1) - re * re
        Eq2 = (x - x2) * (x - x2) + (y - y2) * (y - y2) + (z - z2) * (z - z2) - re * re
        Eq3 = (x - x3) * (x - x3) + (y - y3) * (y - y3) + (z - z3) * (z - z3) - re * re
        if all(-0.001 < Eq < 0.001 for Eq in (Eq1, Eq2, Eq3)):
            check = 1

    return check


# Tính Px, Py, Pz
# t1, t2, t3 degree unit
def Forward_Kinematic(t1, t2, t3):
    t1 = radians(t1)
    t2 = radians(t2)
    t3 = radians(t3)

    x1 = 0
    y1 = -(f - e) / (2 * sqrt(3)) - rf * cos(t1)
    z1 = -rf * sin(t1)

    x2 = ((f - e) / (2 * sqrt(3)) + rf * cos(t2)) * cos(pi / 6)
    y2 = ((f - e) / (2 * sqrt(3)) + rf * cos(t2)) * sin(pi / 6)
    z2 = -rf * sin(t2)

    x3 = -((f - e) / (2 * sqrt(3)) + rf * cos(t3)) * cos(pi / 6)
    y3 = ((f - e) / (2 * sqrt(3)) + rf * cos(t3)) * sin(pi / 6)
    z3 = -rf * sin(t3)

    w1 = x1 * x1 + y1 * y1 + z1 * z1
    w2 = x2 * x2 + y2 * y2 + z2 * z2
    w3 = x3 * x3 + y3 * y3 + z3 * z3

    d = (y2 - y1) * x3 - (y3 - y1) * x2

    a1 = (z2 - z1) * (y3 - y1) - (z3 - z1) * (y2 - y1)
    a2 = -((z2 - z1) * x3 - (z3 - z1) * x2)
    b1 = -((w2 - w1) * (y3 - y1) - (w3 - w1) * (y2 - y1)) / 2
    b2 = ((w2 - w1) * x3 - (w3 - w1) * x2) / 2

    a = a1 * a1 + a2 * a2 + d * d
    b = 2 * (a1 * b1 + a2 * (b2 - y1 * d) - z1 * d * d)
    c = b1 * b1 + (b2 - y1 * d) * (b2 - y1 * d) + (z1 * z1 - re * re) * d * d

    delta = b * b - 4 * a * c
    if delta < 0:
        P = [0, 0, 0, 0]
    else:
        Pz = -0.5 * (b + sqrt(delta)) / a
        Px = (a1 * Pz + b1) / d
        Py = (a2 * Pz + b2) / d

        if check_solution(Px, Py, Pz, x1, y1, z1, x2, y2, z2, x3, y3, z3):
            Pz = Pz + dz
            P = [1, Px, Py, Pz]
        else:
            P = [0, 0, 0, 0]

        P[1] = round(P[1], 2)
        P[2] = round(P[2], 2)
        P[3] = round(P[3], 2)

    return P


def calculate_angle(x0, y0, z0):
    Fi = (-f / (2 * sqrt(3)), 0)

    y1 = Fi[0]
    z1 = Fi[1]

    y2 = y0 - e / (2 * sqrt(3))
    z2 = z0
    r2 = sqrt(re * re - x0 * x0)

    c1 = y1 * y1 + z1 * z1 - rf * rf
    c2 = y2 * y2 + z2 * z2 - r2 * r2
    a = (c1 - c2) / (2 * (z1 - z2))
    b = -(y1 - y2) / (z1 - z2)
    A = b * b + 1
    B = a * b - y1 - z1 * b
    C = a * a - 2 * a * z1 + c1
    delta_ = B * B - A * C

    if delta_ < 0:
        T = [0, 0]
    else:
        yj = (-B - sqrt(delta_)) / A
        zj = a + b * yj
        theta_i = atan2(-zj, (y1 - yj))
        theta_i = degrees(theta_i)
        theta_i = round(theta_i, 2)
        T = [1, theta_i]

    return T


def Inverse_Kinematic(Px, Py, Pz):
    Pz = Pz -dz
    # Ban đầu cho vô nghiệm status = 0
    status = 0
    status1 = calculate_angle(Px, Py, Pz)
    if status1[0]:
        theta1 = status1[1]

        Px2 = Px * cos(-2 * pi / 3) - Py * sin(-2 * pi / 3)
        Py2 = Px * sin(-2 * pi / 3) + Py * cos(-2 * pi / 3)

        status2 = calculate_angle(Px2, Py2, Pz)
        if status2[0]:
            theta2 = status2[1]

            Px3 = Px * cos(2 * pi / 3) - Py * sin(2 * pi / 3)
            Py3 = Px * sin(2 * pi / 3) + Py * cos(2 * pi / 3)

            status3 = calculate_angle(Px3, Py3, Pz)
            if status3[0]:
                theta3 = status3[1]
                status = 1

    if status == 0:
        theta1 = 0
        theta2 = 0
        theta3 = 0

    return [status, theta1, theta2, theta3]

def write_line_data(begin, end, step):
    # Tạo danh sách các giá trị của x 
    x_values = list(range(begin, end + step, step))
    
    # Tính góc theta1, theta2, theta3 cho mỗi giá trị x
    P = [Inverse_Kinematic(x, 0, -450) for x in x_values]
    
    file_path = os.path.join('Gui_Delta', 'line_data.txt')
    # Ghi dữ liệu vào tệp
    with open(file_path, 'w') as file:
        # Thêm dòng đầu tiên
        file.write("theta1,theta2,theta3,Px,Py,Pz\n")
        
        for theta, x in zip(P, x_values):
            file.write(f"{theta[1]},{theta[2]},{theta[3]},{x},0,-450\n")

def read_line_data():
    # Đọc dữ liệu từ tệp
    file_path = os.path.join('Gui_Delta', 'line_data.txt')
    with open(file_path, 'r') as file:
        # Bỏ qua dòng đầu tiên trong file
        header = next(file)
        
        # Đọc và in dữ liệu
        for line in file:
            theta1, theta2, theta3, Px, Py, Pz = map(float, line.strip().split(','))
            yield theta1, theta2, theta3, Px, Py, Pz
            time.sleep(0.5)  # Đặt độ trễ 200ms

# Hàm in giá trị từ read_line_data()
def print_values():
    for theta1, theta2, theta3, Px, Py, Pz in read_line_data():
        print(f"Received values: theta1={theta1}, theta2={theta2}, theta3={theta3}, Px={Px}, Py={Py}, Pz={Pz}")

# write_line_data(-100, 100, 5)
# # Tạo các luồng
# print_values_thread = threading.Thread(target=print_values)
# print_values_thread.start()
# # print_values()
# # code dưới đây để check nhanh động học
# time.sleep(2)
# status, P1, P2, P3 = Forward_Kinematic(-90, -90, -90)
# print(status, P1, P2, P3)
# time.sleep(0.5)
# status1, th1, th2, th3 = Inverse_Kinematic(P1, P2, P3)
# print(status1, th1, th2, th3)
# print_values_thread.join()

