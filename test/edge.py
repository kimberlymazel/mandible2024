import os
import cv2 as cv
import numpy as np

threshold1 = 0

threshold2 = 0


def set_threshold1(t1):
    global threshold1
    threshold1 = t1
def set_threshold2(t2):
    global threshold2
    threshold2 = t2

cv.namedWindow("Limit Selector")

cv.createTrackbar("Threshold 1", "Limit Selector", threshold1, 255, set_threshold1)
cv.createTrackbar("Threshold 2", "Limit Selector", threshold2, 255, set_threshold2)

img_path = os.path.join(os.getcwd(), "dataset", "box", "box_original") # Edit this to target different folder
images = os.listdir(img_path)

img = cv.imread(os.path.join(img_path, images[2])) # Edit images index to get different image

if img.shape[1] > 1080 or img.shape[0] > 1920:
    img = cv.resize(img, (img.shape[1] // 3, img.shape[0] // 3))

while True:
    edges = cv.Canny(img, threshold1, threshold2)
    
    cv.imshow("Limit Selector", edges)

    if cv.waitKey(1) & 0xff == ord('q'): # press q key to exit
        break
    
cv.destroyAllWindows()