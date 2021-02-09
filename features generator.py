import os
import time
import cv2
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)

def corner_count(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    
    dst = cv2.dilate(dst,None)
    x = len(dst)
    
    img[dst>0.01*dst.max()]=[0,0,255]
    red = np.array([0,0,255])
    x = cv2.inRange(img, red, red)
    no = cv2.countNonZero(x)
    return no

def integral_pro(img):
    trans = cv2.transpose(img)
    hp = np.array(0)
    vp = np.array(0)
    for i in img:
        hp = np.append(hp, sum(i))
    for j in trans:
        vp = np.append(vp, sum(j))
    return hp, vp


t0 = time.time()
path = 'C:/Users/krishnaprasad/Desktop/pbi codes/samples'
x = np.array(0)
for entry in os.scandir(path):
    if entry.is_file():
        filename = entry.name
        img = cv2.imread(path + '/' + filename)
        c_no = corner_count(img)
        hrpro, vrpro = integral_pro(img)
        x = np.append(x, c_no)
        x = np.append(x, integral_pro(img))
    print(len(x))
    df = pd.DataFrame(x)
    df.to_csv("features.csv")
print(time.time() - t0)
