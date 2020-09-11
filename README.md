# **中文**

## Kalman filter-PoseNet
此專案的目標是以使用Kalman filter對PoseNet進行影像濾波，是應用於Coral Edge Edge TPU上的PoseNet模型，並呈現出應用於網路攝影機的結果。此專案僅限於單人使用。

PoesNet是一種視覺定位模型，它可以透過一張圖像可以定位身體位置訊息特性是速度較快但精準度不高，所以需要透過濾波測試修正其誤差值，本文以降低雜訊為目地，先將關節座標找出，找出座標點後將座標資料帶入卡爾曼濾波器，進行影像後處理，保留演算法計算出的合理座標。

## 入門
  * 需要在具有Python3.x環境下編譯
  * 此專案開發和測試是在Linux上使用Python完成
  * 使用google-coral內提供的[PoseNet](https://github.com/google-coral/project-posenet.git)專案

### 環境架設

 安裝OpenCV套件
```
sudo apt-get install build-essential cmake unzip pkg-config
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libgtk-3-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install python3-dev
```
 安裝Git:

```
sudo apt-get update
sudo apt-get install git
```
### 檢查OpenCV是否有效

 1. 從此存儲庫下載預構建的庫。
 ```
 git clone https://github.com/pjalusic/opencv4.1.1-for-google-coral.git
 ```
 2. 將cv2.so文件複製到/usr/local/lib/python3.7/dist-packages/中
 ```
cp opencv4.1.1-for-google-coral/cv2.so /usr/local/lib/python3.7/dist-packages/cv2.so 
 ```
 3. 將其他.so文件複製到/ usr / local / lib /中
 ```
sudo cp -r opencv4.1.1-for-google-coral/libraries/. /usr/local/lib 
 ```
 4. 檢查是否有效
 ```
python3
 >>> import cv2
 >>> cv2.__version__
 '4.1.1'
 ```
 
## Kalman filter 介紹

### 優點
 1. 占用記憶體空間少
 2. 適合應用於連續變化的系統下
 3. 適合應用於嵌入式系統
 4. 實現容易，純時域的濾波器

### 原理
  卡爾曼濾波器的主要步驟有兩個：
   1. 預估：濾波器使用上一狀態的估計，做出對當前狀態的估計。
   2. 更新：濾波器利用對當前狀態的觀測值優化在預測階段獲得的預測值，以獲得一個更精確的新估計值。
 ```
 時間更新方程式:           
 
    x_k=A_(x_k−1)+B_uk+Q 
    
    z_k=Hx_k+R	
 ```   
 ```   
 狀態更新方程式:
 
    x_k=x_k−1+Q
    
    z_k=x_k+R				
    
    k_k=P_(k−1)/(P_k+R)
 ```
 A、B、H都是系統參數，在多維的情形下都為矩陣，在簡單的場景下將他們設定為常數1，則公式會簡化成：
 ```
 x_k=x_k−1+Q					
 
 z_k=x_k+R					
 
 k_k=P_(k−1)/(P_k+R)	
 
 x_k=x_(k−1)+k_k (z_k−x_(k−1))			
 
 𝑃_𝑘=（１−k_k）P_(k−1)	

 ```
 ### 比較
 進行卡爾曼濾波器進行影像穩定實驗前，需要先測試我們對於此濾波器的認知是否正確，先將卡爾曼濾波器公式套入Excel內，並將我們影像輸出座標輸入其中，我們使用30fps拍攝180秒影片進行濾波測試如下圖所示，取中間雜訊較為明顯之區段做顯示， 可以發現確實將雜訊濾除並改善了延遲上的問題。
![image](https://github.com/Zhan-KJ/Kalman-filter-PoseNet/blob/master/image/image.png?raw=true)

## 運行方法

### 主要運行方法為下列所述：

 1. 將RGB圖像經過卷積神經網路運算。
 2. 以下列的範例所示，我們將EdgeTPU連接WebCam
   * 確認是否能正確顯示各個關節點
   * 再將各個關節座標以不同顏色做區分，讓使用者更直觀辨識
 3. 由於需比較濾波前後差異，放上濾波測試前後比較，此步驟是在EdgeTPU上運行。
 
 下圖範例為將影像加入關節座標後結果展示。
 
![image](https://github.com/Zhan-KJ/Kalman-filter-PoseNet/blob/master/image/output_single_screen.gif?raw=true)

 下圖範例為濾波前後比較。左側為未濾波狀態，關節點抖動特別明顯，右側為將Kalman filter帶入後狀態，可以發現關節點抖動穩定許多。

![image](https://github.com/Zhan-KJ/Kalman-filter-PoseNet/blob/master/image/output_contrast_screen.gif?raw=true)

# **English**

## Kalman filter-PoseNet
瓜
## Getting Started

## Kalman filter
