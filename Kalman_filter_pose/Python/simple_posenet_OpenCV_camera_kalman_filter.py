import cv2
import time
import sys
import os
import re
import numpy as np
from PIL import Image
from pose_engine import PoseEngine
import argparse
import collections
from functools import partial
from kalman import Kalmanfilter,Point_Kalman_process

cap=cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)
kalman=Kalmanfilter()
kalman_process=Point_Kalman_process()
engine = PoseEngine('models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')

def OpenCVText(frame):
    cv2.putText(frame,'Oriental Institute of Technology',(20,460),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)
    cv2.putText(frame,'Oriental Institute of Technology',(20,460),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)
    cv2.putText(frame,'quit \"Q\"',(560,460),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1,cv2.LINE_AA)
    cv2.putText(frame,'Before \"Red keypoint\"',(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
    cv2.putText(frame,'After \"Blue keypoint\"',(20,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)

def DrawKeypointRed(frame,x,y,label):
    cv2.circle(frame,(x,y),5,(0,0,255),-1)
    cv2.putText(frame,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)

def DrawKeypointBlue(frame,x,y,label):
    x=x.astype('float32')
    y=y.astype('float32')
    cv2.circle(frame,(x,y),3,(255,0,0),-1)
    cv2.putText(frame,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)

def Posenet(frame,kalman,kalman_process):
    image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    image.resize((641, 481), Image.NEAREST)
    poses, inference_time = engine.DetectPosesInImage(np.uint8(image))
    for pose in poses:
        for label, keypoint in pose.keypoints.items():
            if keypoint.score < 0.6: continue
            DrawKeypointRed(frame,keypoint.yx[1],keypoint.yx[0],label)
            out_kalman_x,out_kalman_y=kalman_process.do_kalman_filter(keypoint.yx[1],keypoint.yx[0],label)
            DrawKeypointBlue(frame,out_kalman_x,out_kalman_y,label)

def Camera():
    while(True):
        ret,frame=cap.read()
        Posenet(frame,kalman,kalman_process)
        OpenCVText(frame)
        cv2.namedWindow('simple_posenet_OpenCV_camera_kalman_filter',0)
        cv2.resizeWindow('simple_posenet_OpenCV_camera_kalman_filter',1280,720)
        cv2.imshow('simple_posenet_OpenCV_camera_kalman_filter',frame)
        if cv2.waitKey(1)==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():	
    Camera()


if __name__ == '__main__':
    main()
