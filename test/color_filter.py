import os
import cv2 as cv
import numpy as np

lower_hue = 0
lower_sat = 0
lower_val = 0

upper_hue = 255
upper_sat = 255
upper_val = 255

def set_lower_hue(h):
    global lower_hue
    lower_hue = h
def set_lower_sat(s):
    global lower_sat
    lower_sat = s
def set_lower_val(v):
    global lower_val
    lower_val = v

def set_upper_hue(h):
    global upper_hue
    upper_hue = h
def set_upper_sat(s):
    global upper_sat
    upper_sat = s
def set_upper_val(v):
    global upper_val
    upper_val = v

cv.namedWindow("Limit Selector")

cv.createTrackbar("lower_hue", "Limit Selector", lower_hue, 255, set_lower_hue)
cv.createTrackbar("lower_sat", "Limit Selector", lower_sat, 255, set_lower_sat)
cv.createTrackbar("lower_val", "Limit Selector", lower_val, 255, set_lower_val)

cv.createTrackbar("upper_hue", "Limit Selector", upper_hue, 255, set_upper_hue)
cv.createTrackbar("upper_sat", "Limit Selector", upper_sat, 255, set_upper_sat)
cv.createTrackbar("upper_val", "Limit Selector", upper_val, 255, set_upper_val)

img_path = os.path.join(os.getcwd(), "dataset", "annotations") # Edit this to target different folder
images = os.listdir(img_path)

img = cv.imread(os.path.join(img_path, images[0])) # Edit images index to get different image

if img.shape[1] > 1080 or img.shape[0] > 1920:
    img = cv.resize(img, (img.shape[1] // 3, img.shape[0] // 3))

while True:
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_img, (lower_hue, lower_sat, lower_val), (upper_hue, upper_sat, upper_val))
    result = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)
    result = cv.bitwise_or(img, img, mask=mask.astype(np.uint8))

    cv.imshow("Limit Selector", result)

    if cv.waitKey(1) & 0xff == ord('q'): # press q key to exit
        break
    
cv.destroyAllWindows()