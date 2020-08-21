# **中文**

## Kalman filter-PoseNet
此專案的目標是以使用Kalman filter對PoseNet進行影像濾波，是應用於Coral Edge Edge TPU上的PoseNet模型，並呈現出應用於網路攝影機的結果。此專案僅限於單人使用。

PoesNet是一種視覺定位模型，它可以透過一張圖像可以定位身體位置訊息特性是速度較快但精準度不高，所以需要透過濾波測試修正其誤差值，本文以降低雜訊為目地，先將關節座標找出，找出座標點後將座標資料帶入卡爾曼濾波器，進行影像後處理，保留演算法計算出的合理座標。

## 入門
  * 需要在具有Python3.x環境下編譯
  * 此專案開發和測試是在Linux上使用Python完成的
  * 下載google-coral內提供的[PoseNet](https://github.com/google-coral/project-posenet.git)專案

## pose_camera.py
以模型內的pose_camera.py為基底，來實現影像濾波

## Kalman filter

### 優點
 1. 占用記憶體空間少
 2. 適合應用於連續變化的系統下
 3. 適合應用於嵌入式系統
 4. 實現容易，純時域的濾波器

### 運作原理
  * 卡爾曼濾波器的主要步驟有兩個：
   1. 預估:濾波器使用上一狀態的估計，做出對當前狀態的估計。
   2. 更新:濾波器利用對當前狀態的觀測值優化在預測階段獲得的預測值，以獲得一個更精確的新估計值。
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



# **English**

## Kalman filter-PoseNet

## Getting Started

## pose_camera.py
