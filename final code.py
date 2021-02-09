import os
import time
import cv2
import numpy as np

def sharpen_images(img, path ,i):
    ret, thres = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
    cv2.imwrite(path +  "/" +  str(i) + ".jpg", thres)

t0 = time.time()
path = 'C:/Users/krishnaprasad/Desktop/jpg'
l = path + "/landscape"
p = path + "/portrait"
(i, j) = (1, 1)

for entry in os.scandir(path):
    if entry.is_file():
        image = entry.name
        img = cv2.imread(path + "/" + image,0)
        dim = img.shape[0:2]
        if dim[0] < dim[1]:
            sharpen_images(img, l, i)
            try:
                os.mkdir(l)
            except FileExistsError:
                if os.path.exists(l):
                    print("image is sent to landscape")
                else:
                    raise
            i += 1
        else:
            try:
                os.mkdir(p)
            except FileExistsError:
                if os.path.exists(p):
                    os.rename(path +  "/" + image, p +  "/" +  str(j) + ".jpg")
                    print("image is sent to portrait")
                else:
                    raise
            j += 1
    else:
        print("no images")
print(time.time() - t0)
