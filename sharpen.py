import os
import time
import cv2

path = 'C:/Users/krishnaprasad/Desktop/jpg'
for entry in os.scandir(path):
    if entry.is_file():
        file = entry.name
        x = cv2.imread(path + "/" + file, 0)
        ret, img = cv2.threshold(x, 220, 255, cv2.THRESH_BINARY)
        cv2.imwrite(path +  "/" + file, img)
print("Images are Threshold")
