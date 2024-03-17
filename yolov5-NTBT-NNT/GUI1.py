from tkinter import *
from tkinter.font import BOLD
from Kinematic import *
import serial
import threading

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
from time import sleep
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
count = 0
data  = []
counter = 0

theta1= theta2= theta3 = 0
px= py= pz = 0
connect= 0
connect2 = 0
pump = 0
convey = 0


from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

import cv2
@torch.no_grad()


def run(
        weights=ROOT / 'yolov5-master/runs/train/exp8/weights/best.pt',  # model.pt path(s)
        source= ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    
    global ser, counter, px, py, pz, theta1, theta2, theta3
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:

        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
                start_point = (160, 0)
                end_point = (160, 640)
                # cv2.line(im0, start_point, end_point, color=(0, 255, 0), thickness=2)
                # cv2.imshow('TAT1', im0)
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path  = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                #print(det)
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    center_point = round((c1[0] + c2[0]) / 2), round((c1[1] + c2[1]) / 2)
                    center_point_reference = int(round((c1[0] + c2[0]) / 2)), int((round((c1[1] + c2[1]) / 2) - 100)*0.66666) -100
                    center_point_x = center_point_reference[0]
                    center_point_y = center_point_reference[1]
                    circle = cv2.circle(im0, center_point, 5, (0, 255, 0), 2)
                    text_coord = cv2.putText(im0, str(center_point_reference), center_point, cv2.FONT_HERSHEY_PLAIN, 2,
                                             (0, 255, 0))
                    
                    if(center_point_x >150 and center_point_x <170):
                        print(center_point_y)
                        counter += 1
                        px = 0
                        py = -center_point_y
                        pz = -400

                        theta1 = round(I_kinematic(px, py, pz)[0], 2)
                        theta2 = round(I_kinematic(px, py, pz)[1], 2)
                        theta3 = round(I_kinematic(px, py, pz)[2], 2)
                        print(theta1, theta2, theta3)
                        send_theta()
                        sleep(float(0.5))
                    

                    # if center_point[0] > 320 :

                        ###############################################################################################################
                    #    print(center_point,'Mot vat')
                    # pts = np.array([[c1[0],c1[1]],[c1[0],c2[1]],[c2[0],c1[1]],[c2[0],c2[1]]])
                    # pts = pts.reshape((-1, 1, 2))
                    # cv2.polylines(im0, [pts], True, (0, 0, 255), 3)
                    

                    # Stream results
                    im0 = annotator.result()
                    if view_img:
                        if platform.system() == 'Linux' and p not in windows:
                            windows.append(p)
                            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                        # camthat = cv2.imshow('TAT', im0)  #show cam
                        TAT_img = cv2.resize(im0, (200, 200))
                        cropped_image = im0[100:400, 0: 570]
                        cv2.line(cropped_image, start_point, end_point, color=(0, 255, 0), thickness=2)
                        cv2.imshow('TAT_crop', cropped_image)
                        # print(im0.shape)
                        # print(cropped_imgae.shape)
                        
                        cv2.waitKey(1)  # 1 millisecond

                    if save_txt:  # Write to file

                        c1,c2= (int(xyxy[0]),int(xyxy[1])), (int(xyxy[2]),int(xyxy[3]))
                        center_point = round((c1[0]+c2[0])/2), round((c1[1]+c2[1])/2)
                        circle = cv2.circle(im0,center_point,5,(0,255,0),2)
                        # text_coord = cv2.putText(im0,str(center_point),center_point,cv2.FONT_HERSHEY_PLAIN,2,(0,255,0))

                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyWindow("TAT_crop")
            break
        # Print time (inference-only)
        ########################################################################################################################
        # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    
    


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / '1best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / '0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    #print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

#================================================================================== function for GUI ================================================================

def test_mode():
    global connect
    if(connect == 1):
        cmd = '20 20 20,40 40 40,0 40 40,0 0 40,0 0 0,2' + '\n' 
        ser.write(cmd.encode())

def Classify_mode():
    thread1 = threading.Thread(target=main(parse_opt()), args=())
    thread1.start()

def control_pump():
    global pump
    pump = 1 -pump
    if connect2 == 1:
        if pump == 1:
            cmd_pump = 'a' 
            ser2.write(cmd_pump.encode())
            btn_pump['text'] = 'Pump ON'
            btn_pump['bg'] = 'orange'
        else:
            cmd_pump = 'b'
            ser2.write(cmd_pump.encode())
            btn_pump['text'] = 'Pump OFF'
            btn_pump['bg'] = 'green'
    print(f"pump: {pump}")

def control_convey():
    global convey
    convey = 1 - convey
    if connect2 == 1:
        if convey == 1:
            cmd_convey = 'c'
            ser2.write(cmd_convey.encode())
            btn_conveyor['text'] = 'Pump ON'
            btn_conveyor['bg'] = 'orange'
        else:
            cmd_convey = 'd' 
            ser2.write(cmd_convey.encode())
            btn_conveyor['text'] = 'Pump OFF'
            btn_conveyor['bg'] = 'green'
    print(f"conveyor: {convey}")

def connect_UART():
    global connect, ser
    connect = 1 - connect
    if(connect % 2 == 1):
        port = et_portName.get()
        baud = 115200
        try:
            ser = serial.Serial(port, baud, timeout=1)
        except Exception:
            connect = 0
            print("Serial Port is not define")

        if ser.isOpen():
            print(ser.name + ' is open...')
            btn_connect['text'] = 'Disconnect'
            btn_connect['bg'] = 'orange'

        # thread1 = threading.Thread(target=main(parse_opt()), args=())
        # thread1.start()
    else:
        ser.close()
        btn_connect['text'] = 'Connect'
        btn_connect['bg'] = 'green'


def connect_UART2():
    global connect2, ser, ser2
    connect2 = 1 - connect2
    if(connect2 % 2 == 1):
        port2 = et_portName2.get()
        baud2 = 115200
        try:
            ser2= serial.Serial(port2, baud2, timeout=1)
        except Exception:
            connect2 = 0
            print("Serial2 Port is not define")

        if ser2.isOpen():
            print(ser2.name + ' is open...')
            btn_connect2['text'] = 'Disconnect'
            btn_connect2['bg'] = 'orange'
    else:
        ser2.close()
        btn_connect2['text'] = 'Connect'
        btn_connect2['bg'] = 'green'


def send_theta():
    global connect
    if(connect == 1):
        cmd = str(int(theta1)) + ' '+ str(int(theta2)) + ' '+ str(int(theta3)) + '1\n' 
        ser.write(cmd.encode())


def slider(self):
    theta1= scale_theta_1.get()
    theta2= scale_theta_2.get()
    theta3= scale_theta_3.get()

    entry_theta_1.delete(0,END)
    entry_theta_1.insert(0, str(round(theta1,2)))
    entry_theta_2.delete(0,END)
    entry_theta_2.insert(0, str(round(theta2,2)))
    entry_theta_3.delete(0,END)
    entry_theta_3.insert(0, str(round(theta3,2)))

def F_to_I():
    global px, py, pz, theta1, theta2, theta3
    theta1 = float(entry_theta_1.get())
    theta2 = float(entry_theta_2.get())
    theta3 = float(entry_theta_3.get())
    px = round(F_Kinematic(theta1, theta2, theta3)[0], 2)
    py = round(F_Kinematic(theta1, theta2, theta3)[1], 2)
    pz = round(F_Kinematic(theta1, theta2, theta3)[2], 2)
    scale_theta_1.set(theta1)
    scale_theta_2.set(theta2)
    scale_theta_3.set(theta3)

    et_x_entry.delete(0,END)
    et_x_entry.insert(0,str(round(px,3)))
    et_y_entry.delete(0,END)
    et_y_entry.insert(0,str(round(py,3)))
    et_z_entry.delete(0,END)
    et_z_entry.insert(0,str(round(pz,3)))
    print(px, py, pz)
    send_theta()
    
def I_to_F():
    global px, py, pz, theta1, theta2, theta3
    px= round(float(et_x_entry.get()), 2)
    py= round(float(et_y_entry.get()), 2)
    pz= round(float(et_z_entry.get()), 2)
    theta1 = round(I_kinematic(px, py, pz)[0], 2)
    theta2 = round(I_kinematic(px, py, pz)[1], 2)
    theta3 = round(I_kinematic(px, py, pz)[2], 2)
    print(theta1, theta2, theta3)
    send_theta()
    
def defaul():
    global px, py, pz, theta1, theta2, theta3
    theta1= theta2= theta3 = 0
    entry_theta_1.delete(0,END)
    entry_theta_1.insert(0,str(round(theta1,3)))
    entry_theta_2.delete(0,END)
    entry_theta_2.insert(0,str(round(theta2,3)))
    entry_theta_3.delete(0,END)
    entry_theta_3.insert(0,str(round(theta3,3)))

    px = round(F_Kinematic(theta1, theta2, theta3)[0], 2)
    py = round(F_Kinematic(theta1, theta2, theta3)[1], 2)
    pz = round(F_Kinematic(theta1, theta2, theta3)[2], 2)
    scale_theta_1.set(theta1)
    scale_theta_2.set(theta2)
    scale_theta_3.set(theta3)

    et_x_entry.delete(0,END)
    et_x_entry.insert(0,str(round(px,3)))
    et_y_entry.delete(0,END)
    et_y_entry.insert(0,str(round(py,3)))
    et_z_entry.delete(0,END)
    et_z_entry.insert(0,str(round(pz,3)))
    send_theta()

GUI = Tk()
GUI.title("Delta Robot GUI")
GUI.geometry("800x300")
GUI.resizable(width= False, height= False)

# ===================================================== Device Widget =========================================================
frm_device = Frame(GUI, width = 300, height= 100, highlightbackground= 'yellow', highlightthickness= 2)
frm_device.place(x= 450, y = 0)

frm_title_kinematic = Label(frm_device, text= "CONTROL DEVICE",font= ('Time New Roman',10, BOLD), width= 38, bg= 'yellow')
frm_title_kinematic.pack(side= TOP)

btn_pump= Button(frm_device, width=34,height= 1 , text= "Pump OFF", font=('Time New Roman',10,BOLD),bg= 'green' , command= control_pump)
btn_pump.pack(side= TOP, padx= 1, pady= 1)

btn_conveyor= Button(frm_device, width=34,height= 1 , text= "Conveyor OFF", font=('Time New Roman',10,BOLD),bg= 'green' , command= control_convey)
btn_conveyor.pack(side= TOP, padx= 1, pady= 1)

btn_classify= Button(frm_device, width=34,height= 1 , text= "Classify ON", font=('Time New Roman',10,BOLD),bg= 'green' , command= Classify_mode)
btn_classify.pack(side= TOP, padx= 1, pady= 1)

btn_test= Button(frm_device, width=34,height= 1 , text= "test", font=('Time New Roman',10,BOLD),bg= 'green' , command=test_mode)
btn_test.pack(side= TOP, padx= 1, pady= 1)


# ===================================================== Kinematic Widget =========================================================
frm_kinematic = Frame(GUI, width = 400, height= 400, highlightbackground= 'yellow', highlightthickness= 2)
frm_kinematic.place(x= 10, y = 0)

frm_title_kinematic = Label(frm_kinematic, text= "CONTROL KINEMATIC",font= ('Time New Roman',10, BOLD), width= 52, bg= 'yellow')
frm_title_kinematic.pack(side= TOP)

# ============================================== Forward Widget ==============================================
frm_theta_1 = Frame(frm_kinematic, width = 400, height= 100)
frm_theta_1.pack(side= TOP)
label_theta_1 = Label(frm_theta_1, text= "Theta1")
label_theta_1.pack(side= LEFT)
scale_theta_1 = Scale( frm_theta_1, from_ = 0, to = 120, resolution=0.001, showvalue= 0, orient = HORIZONTAL, length= 300, width= 20, command= slider)
scale_theta_1.pack(side= LEFT)
entry_theta_1 = Entry(frm_theta_1, width= 10, font= ('Time New Roman',10, BOLD), textvariable= "0")
entry_theta_1.pack(side= LEFT)

frm_theta_2 = Frame(frm_kinematic, width = 400, height= 100)
frm_theta_2.pack(side= TOP)
label_theta_2 = Label(frm_theta_2, text= "Theta2")
label_theta_2.pack(side= LEFT)
scale_theta_2 = Scale(frm_theta_2, from_ = 0, to = 120, resolution=0.001, showvalue= 0, orient = HORIZONTAL, length= 300, width= 20, command= slider)
scale_theta_2.pack(side= LEFT)
entry_theta_2 = Entry(frm_theta_2, width= 10, font= ('Time New Roman',10, BOLD))
entry_theta_2.pack(side= LEFT)

frm_theta_3 = Frame(frm_kinematic, width = 400, height= 100)
frm_theta_3.pack(side= TOP)
label_theta_3 = Label(frm_theta_3, text= "Theta3")
label_theta_3.pack(side= LEFT)
scale_theta_3 = Scale(frm_theta_3, from_ = 0, to = 120, resolution=0.001, showvalue= 0, orient = HORIZONTAL, length= 300, width= 20, command= slider)
scale_theta_3.pack(side= LEFT)
entry_theta_3 = Entry(frm_theta_3, width= 10, font= ('Time New Roman',10, BOLD))
entry_theta_3.pack(side= LEFT)

# ============================================== Transfer Widget ==============================================
frm_transfer = Frame(frm_kinematic, width = 400, height= 100)
frm_transfer.pack(side= TOP, pady= 10)
F_to_I_btn = Button(frm_transfer, width=7,text= "Forward", font=('Time New Roman',10,BOLD), command= F_to_I)
F_to_I_btn.pack(side= LEFT)
I_to_F_btn = Button(frm_transfer, width=7,text= "Inverse", font=('Time New Roman',10,BOLD), command= I_to_F)
I_to_F_btn.pack(side= LEFT)
default_btn = Button(frm_transfer, width=7,text= "Default", font=('Time New Roman',10,BOLD), command= defaul)
default_btn.pack(side= LEFT)

# ============================================== Inverse Widget ===============================================
frm_position = Frame(frm_kinematic, width = 400, height= 100)
frm_position.pack(side= TOP, pady= 10)

frm_position_x = Frame(frm_position, width = 400, height= 100)
frm_position_x.pack(side= LEFT, padx= 10, pady= 10)
lb_position_x = Label(frm_position_x,text= "PX")
lb_position_x.pack(side= TOP)
et_x_entry = Entry(frm_position_x, width= 10, font= ('Time New Roman',10, BOLD))
et_x_entry.pack(side= TOP)

frm_position_y = Frame(frm_position, width = 400, height= 100)
frm_position_y.pack(side= LEFT, padx=10, pady= 10)
lb_position_y = Label(frm_position_y,text= "PY")
lb_position_y.pack(side= TOP)
et_y_entry = Entry(frm_position_y, width= 10, font= ('Time New Roman',10, BOLD))
et_y_entry.pack(side= TOP)

frm_position_z = Frame(frm_position, width = 400, height= 100)
frm_position_z.pack(side= LEFT, padx=10, pady= 10)
lb_position_z = Label(frm_position_z,text= "PZ")
lb_position_z.pack(side= TOP)
et_z_entry = Entry(frm_position_z, width= 10, font= ('Time New Roman',10, BOLD))
et_z_entry.pack(side= TOP)



# ===================================================== Connect Widget =========================================================
frm_connect = Frame(GUI, width = 100, height= 100, highlightbackground= 'yellow', highlightthickness= 2)
frm_connect.place(x = 450, y = 160)
frm_connect1 = Frame(frm_connect)
frm_connect1.pack(side= TOP)
frm_connect2 = Frame(frm_connect)
frm_connect2.pack(side= TOP)

# ====================================================== Enter Widget1 ==========================================================
frm_title_connect = Label(frm_connect1, text= "CONNECTION",font= ('Time New Roman',10, BOLD), width= 38, bg= 'yellow')
frm_title_connect.pack(side= TOP)

frm_enter_connect= Frame(frm_connect1)
frm_enter_connect.pack(side= TOP)
lb_connect= Label(frm_enter_connect, text= "DELTA_PORT:  ")
lb_connect.pack(side= LEFT, padx= 10, pady= 10)
et_portName= Entry(frm_enter_connect, width= 10, font= ('Time New Roman',10, BOLD))
et_portName.pack(side= LEFT, padx= 10, pady= 10)
et_portName.insert(0,'COM12')
btn_connect= Button(frm_enter_connect, width=9,height= 1 , text= "Connect", font=('Time New Roman',10,BOLD),bg= 'green' , command= connect_UART)
btn_connect.pack(side= LEFT, padx= 10, pady= 10)

# ====================================================== Enter Widget2 ==========================================================
frm_enter_connect2= Frame(frm_connect2)
frm_enter_connect2.pack(side= TOP)
lb_connect2= Label(frm_enter_connect2, text= "DEVICE_PORT:")
lb_connect2.pack(side= LEFT, padx= 10, pady= 10)
et_portName2= Entry(frm_enter_connect2, width= 10, font= ('Time New Roman',10, BOLD))
et_portName2.pack(side= LEFT, padx= 10, pady= 10)
et_portName2.insert(0,'COM8')
btn_connect2= Button(frm_enter_connect2, width=9,height= 1 , text= "Connect", font=('Time New Roman',10,BOLD),bg= 'green' , command= connect_UART2)
btn_connect2.pack(side= LEFT, padx= 10, pady= 10)


defaul()
GUI.mainloop()