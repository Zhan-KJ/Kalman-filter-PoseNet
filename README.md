# **中文**

## Kalman filter-PoseNet
此專案的目標是以使用Kalman filter對PoseNet進行影像濾波，是應用於Coral Edge Edge TPU上的PoseNet模型，並呈現出應用於網路攝影機的結果。此專案僅限於單人使用。

PoesNet是一種視覺定位模型，它可以透過一張圖像可以定位身體位置訊息特性是速度較快但精準度不高，所以需要透過濾波測試修正其誤差值，本文以降低雜訊為目地，先將關節座標找出，找出座標點後將座標資料帶入卡爾曼濾波器，進行影像後處理，保留演算法計算出的合理座標。

### 優點
 1.占用記憶體空間少
 2.適合應用於連續變化的系統下
 3.適合應用於嵌入式系統
 4.實現容易，純時域的濾波器


## 入門
  * 需要在具有Python3.x環境下編譯
  * 此專案開發和測試是在Linux上使用Python完成的
  * 下載google-coral內提供的[PoseNet](https://github.com/google-coral/project-posenet.git)專案

## pose_camera.py
以模型內的pose_camera.py為基底，來實現影像濾波





# **English**

## Kalman filter-PoseNet

## Getting Started

## pose_camera.py
