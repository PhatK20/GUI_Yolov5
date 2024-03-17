import math
import time
start = 0
t = 0
u = 0
def trajectory_planing():
    global start,t ,u
    end = time.time_ns()//1000000
    if (end-start>=20): #20ms
        start = time.time_ns()//1000000
        t = t + 20
        if (t>1.5*1000):
            t = 0
            u = 1
        tf = 1.5*1000 # thoi gian di chuyen giua 2 diem
        t1 = 1.5*1000 # thoi gian di ngang
        t2 = 3*1000 # thoi gian di xuong hut
        t3 = 4.5*1000 # thoi gian di len
        t4 = 6*1000 # thoi gian phan loai
        t5 = 6.1*1000 # thoi gian nha
        t6 = 7.6*1000 # thoi gian tro ve
        Px1 = 0
        Py1 = 0
        Pz1 = -294.54
        Px2 = 0
        Py2 = -50 #centerpoint
        Pz2 = -323.46
        Px3 = 0
        Py3 = -50 #centerpoint
        Pz3 = -400
        Px4 = 0
        Py4 = -50 #centerpoint
        Pz4 = -323.46
        ########################################### (t<t1)##########################################################
        if (t<t1): 
            q0 = Px1
            qf = Px2
            v0 = 0
            vf =0 
            a00 = 0
            a0f = 0
            a0 = Px1
            a1 = v0
            a2 = 0.5*a00
            a3 = -(20*q0-20*qf+12*tf*v0+8*tf*vf+3*a00*tf**2-a0f*tf**2)/(2*tf**3)
            a4 = (30*q0-30*qf+16*tf*v0+14*tf*vf+3*a00*tf**2-a0f*tf**2)/(2*tf**4)
            a5 = -(12*q0-12*qf+6*tf*v0+6*tf*vf+a00*tf**2-a0f*tf**2)/(2*tf**5)
            Px = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5 
            
            q01 = Py1#Py2
            qf1 = Py2#Py3
            v0 = 0
            vf =0 
            a00 = 0
            a0f = 0
            a01 = q01
            a11 = v0
            a21 = 0.5*a00
            a31 = -(20*q01-20*qf1+12*tf*v0+8*tf*vf+3*a00*tf**2-a0f*tf**2)/(2*tf**3)
            a41 = (30*q01-30*qf1+16*tf*v0+14*tf*vf+3*a00*tf**2-2*a0f*tf**2)/(2*tf**4)
            a51 = -(12*q01-12*qf1+6*tf*v0+6*tf*vf+a00*tf**2-a0f*tf**2)/(2*tf**5)
            Py = a01 + a11*t + a21*t**2 + a31*t**3 + a41*t**4 + a51*t**5

            q02 = Pz1#
            qf2 = Pz2
            v0 = 0
            vf =0 
            a00 = 0
            a0f = 0
            a02 = q02
            a12 = v0
            a22 = 0.5*a00
            a32 = -(20*q02-20*qf2+12*tf*v0+8*tf*vf+3*a00*tf**2-a0f*tf**2)/(2*tf**3)
            a42 = (30*q02-30*qf2+16*tf*v0+14*tf*vf+3*a00*tf**2-2*a0f*tf**2)/(2*tf**4)
            a52 = -(12*q02-12*qf2+6*tf*v0+6*tf*vf+a00*tf**2-a0f*tf**2)/(2*tf**5)
            Pz = a02 + a12*t + a22*t**2 + a32*t**3 + a42*t**4 + a52*t**5
        else:
            Px = 0
            Py = -50
            Pz = -323.46
        ########################################### (t>t1&t<t2)##########################################################
        if (t>t1&t<t2): 
            q0 = Px2
            qf = Px3
            v0 = 0
            vf =0 
            a00 = 0
            a0f = 0
            a0 = Px2
            a1 = v0
            a2 = 0.5*a00
            a3 = -(20*q0-20*qf+12*tf*v0+8*tf*vf+3*a00*tf**2-a0f*tf**2)/(2*tf**3)
            a4 = (30*q0-30*qf+16*tf*v0+14*tf*vf+3*a00*tf**2-a0f*tf**2)/(2*tf**4)
            a5 = -(12*q0-12*qf+6*tf*v0+6*tf*vf+a00*tf**2-a0f*tf**2)/(2*tf**5)
            Px = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5 
            
            q01 = Py2
            qf1 = Py3
            v0 = 0
            vf =0 
            a00 = 0
            a0f = 0
            a01 = q01
            a11 = v0
            a21 = 0.5*a00
            a31 = -(20*q01-20*qf1+12*tf*v0+8*tf*vf+3*a00*tf**2-a0f*tf**2)/(2*tf**3)
            a41 = (30*q01-30*qf1+16*tf*v0+14*tf*vf+3*a00*tf**2-2*a0f*tf**2)/(2*tf**4)
            a51 = -(12*q01-12*qf1+6*tf*v0+6*tf*vf+a00*tf**2-a0f*tf**2)/(2*tf**5)
            Py = a01 + a11*t + a21*t**2 + a31*t**3 + a41*t**4 + a51*t**5

            q02 = Pz2
            qf2 = Pz3
            v0 = 0
            vf =0 
            a00 = 0
            a0f = 0
            a02 = q02
            a12 = v0
            a22 = 0.5*a00
            a32 = -(20*q02-20*qf2+12*tf*v0+8*tf*vf+3*a00*tf**2-a0f*tf**2)/(2*tf**3)
            a42 = (30*q02-30*qf2+16*tf*v0+14*tf*vf+3*a00*tf**2-2*a0f*tf**2)/(2*tf**4)
            a52 = -(12*q02-12*qf2+6*tf*v0+6*tf*vf+a00*tf**2-a0f*tf**2)/(2*tf**5)
            Pz = a02 + a12*t + a22*t**2 + a32*t**3 + a42*t**4 + a52*t**5
        else:
            Px = 0
            Py = -50
            Pz = -323.46
            
        ########################################### (t>t2&t<t3)##########################################################
        if (t>t2&t<t3): 
            q0 = Px2
            qf = Px3
            v0 = 0
            vf =0 
            a00 = 0
            a0f = 0
            a0 = Px2
            a1 = v0
            a2 = 0.5*a00
            a3 = -(20*q0-20*qf+12*tf*v0+8*tf*vf+3*a00*tf**2-a0f*tf**2)/(2*tf**3)
            a4 = (30*q0-30*qf+16*tf*v0+14*tf*vf+3*a00*tf**2-a0f*tf**2)/(2*tf**4)
            a5 = -(12*q0-12*qf+6*tf*v0+6*tf*vf+a00*tf**2-a0f*tf**2)/(2*tf**5)
            Px = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5 
            
            q01 = Py2
            qf1 = Py3
            v0 = 0
            vf =0 
            a00 = 0
            a0f = 0
            a01 = q01
            a11 = v0
            a21 = 0.5*a00
            a31 = -(20*q01-20*qf1+12*tf*v0+8*tf*vf+3*a00*tf**2-a0f*tf**2)/(2*tf**3)
            a41 = (30*q01-30*qf1+16*tf*v0+14*tf*vf+3*a00*tf**2-2*a0f*tf**2)/(2*tf**4)
            a51 = -(12*q01-12*qf1+6*tf*v0+6*tf*vf+a00*tf**2-a0f*tf**2)/(2*tf**5)
            Py = a01 + a11*t + a21*t**2 + a31*t**3 + a41*t**4 + a51*t**5

            q02 = Pz2
            qf2 = Pz3
            v0 = 0
            vf =0 
            a00 = 0
            a0f = 0
            a02 = q02
            a12 = v0
            a22 = 0.5*a00
            a32 = -(20*q02-20*qf2+12*tf*v0+8*tf*vf+3*a00*tf**2-a0f*tf**2)/(2*tf**3)
            a42 = (30*q02-30*qf2+16*tf*v0+14*tf*vf+3*a00*tf**2-2*a0f*tf**2)/(2*tf**4)
            a52 = -(12*q02-12*qf2+6*tf*v0+6*tf*vf+a00*tf**2-a0f*tf**2)/(2*tf**5)
            Pz = a02 + a12*t + a22*t**2 + a32*t**3 + a42*t**4 + a52*t**5
        else:
            Px = 0
            Py = -50
            Pz = -323.46

        theta1 = round(I_kinematic(Px, py, pz)[0], 2)
        theta2 = round(I_kinematic(Px, py, pz)[1], 2)
        theta3 = round(I_kinematic(px, py, pz)[2], 2)
        print("t: ",t)
        print("Px: ", Px)
        print("Py: ", Py)
        print("Pz: ", Pz)
    return u
while True: 
    u=trajectory_planing()
    if (u==1):
        break