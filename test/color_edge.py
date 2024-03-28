import os
import cv2 as cv
import numpy as np
import math

threshold1 = 255
threshold2 = 255

lower_hue = 0
lower_sat = 0
lower_val = 0

upper_hue = 255
upper_sat = 255
upper_val = 255

def set_threshold1(t1):
    global threshold1
    threshold1 = t1
def set_threshold2(t2):
    global threshold2
    threshold2 = t2

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

cv.createTrackbar("Threshold 1", "Limit Selector", threshold1, 255, set_threshold1)
cv.createTrackbar("Threshold 2", "Limit Selector", threshold2, 255, set_threshold2)

cv.createTrackbar("lower_hue", "Limit Selector", lower_hue, 255, set_lower_hue)
cv.createTrackbar("lower_sat", "Limit Selector", lower_sat, 255, set_lower_sat)
cv.createTrackbar("lower_val", "Limit Selector", lower_val, 255, set_lower_val)

cv.createTrackbar("upper_hue", "Limit Selector", upper_hue, 255, set_upper_hue)
cv.createTrackbar("upper_sat", "Limit Selector", upper_sat, 255, set_upper_sat)
cv.createTrackbar("upper_val", "Limit Selector", upper_val, 255, set_upper_val)

img_index = -1


img_path = os.path.join(os.getcwd(), "dataset", "annotations2") # Edit this to target different folder
images = list(filter(lambda p: p.split(".")[0][-1] == "1", os.listdir(img_path)))
image = cv.imread(os.path.join(img_path, images[img_index])) # Edit images index to get different image

# Region of Interest
roi_x1 = round(image.shape[1] * 0.3)
roi_y1 = round(image.shape[0] * 0.3)  
roi_x2 = round(image.shape[1] * 0.7)
roi_y2 = round(image.shape[0] * 0.7)

def is_contour_in_roi(xs, ys):
    x_axis = min(xs) >= roi_x1 and max(xs) < roi_x2
    y_axis = min(ys) >= roi_y1 and max(ys) < roi_y2
    return x_axis and y_axis

while True:
    img = image.copy()
    if img.shape[1] > 1080 or img.shape[0] > 1920:
        img = cv.resize(img, (img.shape[1] // 3, img.shape[0] // 3))        
        
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    mask = cv.inRange(hsv_img, (lower_hue, lower_sat, lower_val), (upper_hue, upper_sat, upper_val))
    
    bgr_img = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)
    bgra_img = cv.cvtColor(hsv_img, cv.COLOR_BGR2BGRA)
    
    bit_img = cv.bitwise_or(img, img, mask=mask.astype(np.uint8))
    
    edges = cv.Canny(bit_img, threshold1, threshold2)
    
    contours, hierarchies = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    
    
    for contours in contours:
        contours = contours.reshape(contours.shape[0], 2)
        xs, ys = contours.T
        if is_contour_in_roi(xs, ys):
            sorted_list = sorted(contours.tolist(), key=lambda point: sum(point))
            
            x_len = max(xs) - min(xs)
            y_len = max(ys) - min(ys)
            
            if y_len > 60:
                cv.circle(img, sorted_list[0], 5, (255, 0, 0), 1)
                cv.circle(img, sorted_list[-1], 5, (0, 0, 255), 1)
                cv.line(img, sorted_list[0], sorted_list[-1], (255, 0, 0), 5)
                x1, y1 = sorted_list[0]
                x2, y2 = sorted_list[-1]
                
                ldx, ldy = max(x2, x1) - min(x2, x1), max(y2, y1) - min(y2, y1)
                line_length = math.sqrt((ldx ** 2) + (ldy ** 2))  
        

    cv.imshow("Limit Selector", img)
    
    if cv.waitKey(1) & 0xff == ord('q'): 
        break
    
cv.destroyAllWindows()