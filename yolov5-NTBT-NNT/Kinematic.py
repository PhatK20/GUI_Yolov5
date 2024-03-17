from math import *

e = 60
f = 110
re = 360
rf = 157

def F_Kinematic(theta1, theta2, theta3):
    t1 = theta1*pi/180
    t2 = theta2*pi/180
    t3 = theta3*pi/180

    y1=-(f-e+rf*cos(t1))
    z1 = -rf*sin(t1)

    x2=(f-e+rf*cos(t2))*cos(pi/6)
    y2=(f-e+rf*cos(t2))*sin(pi/6)
    z2 = -rf*sin(t2)

    x3=-(f-e+rf*cos(t3))*cos(pi/6)
    y3=(f-e+rf*cos(t3))*sin(pi/6)
    z3 = -rf*sin(t3)

    d = (y2-y1)*x3-(y3-y1)*x2
    w1 = y1*y1+z1*z1
    w2 = x2*x2+y2*y2+z2*z2
    w3 = x3*x3+y3*y3+z3*z3

    a1 = ((z2-z1)*(y3-y1)-(z3-z1)*(y2-y1))/d
    a2 = -((z2-z1)*x3-(z3-z1)*x2)/d
    b1 = -((w2-w1)*(y3-y1)-(w3-w1)*(y2-y1))/(2*d)
    b2 = ((w2-w1)*x3-(w3-w1)*x2)/(2*d)

    a = a1*a1+a2*a2+1
    b = 2*(a1+a2*(b2-y1)-z1)
    c = b1*b1+(b2-y1)*(b2-y1)+z1*z1-re*re

    dt = b*b-4*a*c

    if (dt>0):
        pz = (-b-sqrt(dt))/(2*a)
        px = a1*pz+b1
        py = a2*pz+b2
    
    return px, py, pz

def cal_theta(p_x, p_y, p_z):
    x0 = p_x 
    z0 = p_z
    y0 = p_y - e 
    y1 = -f  
    a = (x0*x0 + y0*y0 + z0*z0 +rf*rf - re*re - y1*y1)/(2*z0)
    b = (y1-y0)/z0
    d = -(a+b*y1)*(a+b*y1)+rf*(b*b*rf+rf) 
    if (d > 0):
        yj = (y1 - a*b - sqrt(d))/(b*b + 1)
        zj = a + b*yj
        theta = atan2(-zj,(y1 - yj))*180/pi
    return theta

def I_kinematic(px, py, pz):
    theta1 = cal_theta(px,py,pz)
    px2 = px*cos(2*pi/3)+py*sin(2*pi/3)
    py2 = -px*sin(2*pi/3)+py*cos(2*pi/3)
    theta2 = cal_theta(px2,py2,pz)
    px3 = px*cos(-2*pi/3)+py*sin(-2*pi/3)
    py3 = -px*sin(-2*pi/3)+py*cos(-2*pi/3)
    theta3 = cal_theta(px3,py3,pz)

    return theta1, theta2, theta3
