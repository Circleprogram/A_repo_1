import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from collections import deque
from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing

maxLength = 90
step = 15
trac_queue = deque(maxlen=maxLength)
last_queue = deque(maxlen=step)


video_path = "orange2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
print("fps: ", fps)
t = 1 / fps

kf = cv2.KalmanFilter(6,4)
# kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]], np.float32)
# kf.transitionMatrix = np.array([[1,0,t,0], [0,1,0,t], [0,0,1,0], [0,0,0,1]], np.float32)
kf.measurementMatrix = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0]], np.float32)
kf.transitionMatrix = np.array([[1,0,t,0,(t**2)/2,0], 
                                [0,1,0,t,0,(t**2)/2],
                                [0,0,1,0,t,0],
                                [0,0,0,1,0,t],
                                [0,0,0,0,1,0],
                                [0,0,0,0,0,1]], np.float32)

kf.processNoiseCov = 1e-1 * np.eye(6, dtype=np.float32)  # Q
kf.measurementNoiseCov = 1e-2 * np.eye(4, dtype=np.float32)  # R

frameCounter = 0
opp_frameCounter = 0
center = (1.0, 1.0)
precenter = (1.0, 1.0)
nextstate = [1.0, 1.0, 1,0, 1.0]
vx = 1.0
vy = 1.0
preradius = 1.0
Timelist = []
Xlist = []
Ylist = []
while True:
    ret, frame = cap.read()

    if not ret:
        # break
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frameCounter += 1
        
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsvImage)

    kernel1 = np.ones((5,5), np.uint8)

    erodedImage1 = cv2.erode(S, kernel1, iterations=2)
    dilatedImage1 = cv2.dilate(erodedImage1, kernel1, iterations=2)

    _, binaryImage = cv2.threshold(dilatedImage1, 192, 255, cv2.THRESH_BINARY)

    kernel2  = np.ones((5,5), np.uint8)

    dilatedImage2 = cv2.dilate(binaryImage, kernel2, iterations=5)
    erodedImage2 = cv2.erode(dilatedImage2, kernel2, iterations=5)

    # 橘子轮廓和中心
    contours, hierarchy = cv2.findContours(erodedImage2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        for contour in contours:
            center, radius = cv2.minEnclosingCircle(contour)
            '''
            if frameCounter < 120:
                Timelist.append(frameCounter)
                Xlist.append(center[0])
                Ylist.append(center[1])
            '''
            vx = float((center[0] - precenter[0]) / t)
            vy = float((center[1] - precenter[1]) / t)
            precenter = center
            preradius = radius
            center = (int(center[0]), int(center[1]))
            radius = int(radius)
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            print("frame", frameCounter, " 1",center, vx, vy)
    
    else:
        center = (nextstate[0], nextstate[1])
        vx = float(nextstate[2])
        vy = float(nextstate[3])
        precenter = center
        preradius = radius
        center = (int(center[0]), int(center[1]))
        radius = int(radius)
        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        print("frame", frameCounter, " 2", center, vx, vy)
    
    # 卡尔曼预测下一帧
    measurement = np.array([[center[0]], [center[1]], [vx], [vy]], np.float32)
    prediction = kf.predict()
    kf.correct(measurement)
    nextstate = prediction
    cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 5, (255, 255, 0), -1)

    # Holt-winters预测15帧
    trac_queue.append(center)  # 现有maxLength帧轨迹
    if len(trac_queue) == maxLength:
        list_from_queue = list(trac_queue)
        df_data = pd.DataFrame(list_from_queue, columns=["X", "Y"])
        X = df_data["X"]
        Y = df_data["Y"]

        modelX = ExponentialSmoothing(X, trend='mul', seasonal='add', seasonal_periods=44)
        modelY = ExponentialSmoothing(Y, trend='add', seasonal='add', seasonal_periods=22)

        # 拟合模型
        fit_modelX = modelX.fit()
        fit_modelY = modelY.fit()

        # 预测下一步到后面第15帧
        forecastX = fit_modelX.predict(len(X)+1, len(X)+step)
        forecastY = fit_modelY.predict(len(Y)+1, len(Y)+step)

        posx = forecastX.iloc[-1]
        posy = forecastY.iloc[-1]
        
        if len(last_queue) == step:
            last_posx = last_queue[0][0]
            last_posy = last_queue[0][1]
            cv2.circle(frame, (int(last_posx), int(last_posy)), 5, (0, 0, 255), 2)

        last_queue.append((posx, posy))
        
        # cv2.circle(frame, (int(posx), int(posy)), 5, (0, 0, 255), 2)

        '''
        fig, axs = plt.subplots(1,2)
        axs[0].plot(X, label='Source data')
        axs[0].plot(fit_modelX.fittedvalues, label='Regression data')
        axs[0].plot(range(len(X),len(X)+step), forecastX, 'ro', label='Predictioin')
        axs[0].set_title("X prediction")
        axs[0].legend()

        axs[1].plot(Y, label='Source data')
        axs[1].plot(fit_modelY.fittedvalues, label='Regression data')
        axs[1].plot(range(len(Y),len(Y)+step), forecastY, 'ro', label='Predictioin')
        axs[1].set_title("Y prediction")
        axs[1].legend()

        #plt.tight_layout()
        #plt.figure(figsize=(60, 40))
        plt.show()
        '''

    '''
    if frameCounter == 120:
        new_Timelist = np.linspace(1, 120, num=120)
        new_Xlist = np.interp(new_Timelist, Timelist, Xlist)
        new_Ylist = np.interp(new_Timelist, Timelist, Ylist)
        plt.plot(new_Timelist, new_Xlist, 'o', '-', label='Data X')
        plt.plot(new_Timelist, new_Ylist, 'x', '-', label="Data Y")
        plt.title("Linear-interp Multiple Line Plot")
        plt.xlabel("Time")
        plt.ylabel("Data")
        plt.savefig('newplot.png')
        print("1")
        
        savefile = "data.csv"
        savedata = zip(new_Timelist, new_Xlist, new_Ylist)
        with open(savefile, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Time', 'X', 'Y'])
            csvwriter.writerows(savedata)
        print(f"Data has been written to {savefile}")
    '''

    cv2.imshow('Binary Image', binaryImage)
    cv2.imshow('Result Image', frame)
    
    if cv2.waitKey(30) == 27:
        break;
    
    

cap.release()
cv2.destroyAllWindows()