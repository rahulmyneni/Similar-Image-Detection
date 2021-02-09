import numpy as np
import cv2

img = cv2.imread('name.jpg',0)
dim = img.shape[0:1]

if dim[0] > dim[1]:
    print("image is portrait")
elif dim[0] < dim[1]:
    print("image is landscape")
else:
    print("square image")
