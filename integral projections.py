import numpy as np
import cv2
from matplotlib import pyplot as plt
def pro(img):
    trans = cv2.transpose(img)
    hp = np.array(0)
    vp = np.array(0)
    for i in img:
        hp = np.append(hp, sum(i))
    for j in trans:
        vp = np.append(vp, sum(j))
    return hp, vp



img = cv2.imread('test1.jpg',0)
img2 = cv2.imread('same1.jpg',0)

hrpro, vrpro = pro(img)
hrpro1, vrpro1 = pro(img2)

plt.plot(hrpro)
plt.show()
plt.plot(hrpro1)
plt.show()
plt.plot(vrpro)
plt.show()
plt.plot(vrpro1)
plt.show()
