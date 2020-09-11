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

cap=cv2.VideoCapture('test_video/three_mins_test_video.mp4')
kalman=Kalmanfilter()
kalman_process=Point_Kalman_process()
engine = PoseEngine('models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')

fourcc=cv2.VideoWriter_fourcc(*'XVID')
video_writer=cv2.VideoWriter("output_video/output_contrast_screen.avi",fourcc,30,(1280,480))

def DrawKeypointBlue(imgs,x,y,label):
    x=x.astype('int')+640
    y=y.astype('float32')
    cv2.circle(imgs,(x,y),5,(255,0,0),-1)
    cv2.putText(imgs,str(int(x))+","+str(int(y)),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)

def Posenet(frame,kalman,kalman_process,imgs):
    kalman_process.reset_kalman_filter_X
    kalman_process.reset_kalman_filter_Y
    image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    image.resize((641, 481), Image.NEAREST)
    poses, inference_time = engine.DetectPosesInImage(np.uint8(image))
    for pose in poses:
        for label, keypoint in pose.keypoints.items():
            if keypoint.score < 0.6: continue
            cv2.circle(imgs,(keypoint.yx[1],keypoint.yx[0]),5,(0,0,255),-1)
            cv2.putText(imgs,str(int(keypoint.yx[1]))+","+str(int(keypoint.yx[0])),(keypoint.yx[1],keypoint.yx[0]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1,cv2.LINE_AA)
            out_kalman_x,out_kalman_y=kalman_process.do_kalman_filter(keypoint.yx[1],keypoint.yx[0],label)
            DrawKeypointBlue(imgs,out_kalman_x,out_kalman_y,label)

def OpenCVText(imgs):
    cv2.putText(imgs,'Before',(50,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3,cv2.LINE_AA)
    cv2.putText(imgs,'After',(690,70),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3,cv2.LINE_AA)
    cv2.putText(imgs,'Oriental Institute of Technology',(500,460),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)
    cv2.putText(imgs,'quit \"Q\"',(1200,460),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1,cv2.LINE_AA)

def Video():
    while cap.isOpened():
        ret,frame=cap.read()
        imgs=np.hstack([frame,frame])
        Posenet(frame,kalman,kalman_process,imgs)
        cv2.namedWindow('Demo_three_mins_video_OpenCV_Posenet_Kalman_filter_Contrast',0)
        cv2.resizeWindow('Demo_three_mins_video_OpenCV_Posenet_Kalman_filter_Contrast',1280,480)
        OpenCVText(imgs)
        video_writer.write(imgs)        
        cv2.imshow('Demo_three_mins_video_OpenCV_Posenet_Kalman_filter_Contrast',imgs)
        if cv2.waitKey(33)==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    Video()

if __name__ == '__main__':
    main()
