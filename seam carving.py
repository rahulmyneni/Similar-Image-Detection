from skimage import transform
from skimage import filters
import cv2

image = cv2.imread("test.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

mag = filters.sobel(gray.astype("float"))

cv2.imshow("Original", image)
for numSeams in range(20, 140, 20):
    carved = transform.seam_carve(image, mag, 'horizantal',numSeams)
    print("[INFO] removing {} seams; new size: " "w={}, h={}".format(numSeams, carved.shape[1],carved.shape[0]))
    cv2.imshow("Carved", carved)
    cv2.waitKey(0)
