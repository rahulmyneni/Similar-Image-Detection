import os
import cv2

path = 'C:/Users/krishnaprasad/Desktop/jpg'
l = "landscape"
p = "portrait"
(i, j) = (1, 1)
for entry in os.scandir(path):
    if entry.is_file():
        file = entry.name
        img = cv2.imread(path + "/" + file,0)
        dim = img.shape[0:2]
        if dim[0] < dim[1]:
            sharpen_images(img, l)
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
                os.mkdir(path + "/" + p)
            except FileExistsError:
                if os.path.exists(path + "/" +  p):
                    os.rename(path +  "/" + file, path + "/" + p +  "/" + str(j) + ".jpg")
                    print("sent to " + p)
                else:
                    raise
                j += 1

