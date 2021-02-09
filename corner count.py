import os
import time
import cv2
import numpy as np

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


t0 = time.time()
path = 'C:/Users/krishnaprasad/Desktop/jpg'
l = path + "/landscape"
p = path + "/portrait"

for entry in os.scandir(l):
    if entry.is_file():
        filename = entry.name
        img = cv2.imread(l + '/' + filename)
        corner_count(img)
        print(filename + ' || ' + str(no))

print(time.time() - t0)
