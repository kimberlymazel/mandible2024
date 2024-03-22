import os
import cv2 as cv
import numpy as np

def get_annotation_lines(img: np.array):
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_img, (1, 0, 0), (255, 255, 255))
    return mask

def lines_to_points(mask: np.array):
    top = []
    bottom = []

    for x in range(mask.shape[1]):
        points = np.argwhere(mask[:, x] > 0)
        
        if points.shape[0] > 0:
            bottom.append((x, max(points)[0]))
            
            if max(points) - min(points) > 100:
                top.append((x, min(points)[0]))
                
    top = np.array(top, dtype=np.uint32)
    bottom = np.array(bottom, dtype=np.uint32)
    
    return top, bottom

def get_segmentation(img: np.array, img_shape: tuple[int, int]):
    segmentation = np.zeros(img_shape).astype(np.uint8)
    annotation_mask = get_annotation_lines(img)
    top, bottom = lines_to_points(annotation_mask)
    
    for i in range(0, bottom.shape[0] - 2):
        x1, y1 = bottom[i]
        x2, y2 = bottom[i + 1]
        cv.line(segmentation, (x1, y1), (x2, y2), (255, 255, 255), 1)

    for i in range(0, top.shape[0] - 2):
        x1, y1 = top[i]
        x2, y2 = top[i + 1]
        cv.line(segmentation, (x1, y1), (x2, y2), (255, 255, 255), 1)

    cv.line(segmentation, bottom[0, :], top[0, :], (255, 255, 255), 1)
    cv.line(segmentation, bottom[-1, :], top[-1, :], (255, 255, 255), 1)
    
    gray_segmentation = cv.cvtColor(segmentation, cv.COLOR_BGR2GRAY)
    for x in range(gray_segmentation.shape[1]):
        points = np.argwhere(gray_segmentation[:, x] > 0)
        if points.shape[0] > 0:
            gray_segmentation[min(points)[0]:max(points)[0] + 1, x] = 255
    
    return gray_segmentation

if __name__ == "__main__":
    input_folder_path = os.path.join(os.getcwd(), "dataset", "measurement results and X-rays with lines Merve")
    image_names = list(filter(lambda img: img.split(".")[-1] != "xlsx", os.listdir(input_folder_path)))
    output_folder_path = os.path.join(os.getcwd(), "dataset", "test-segmentation")
    
    for img_name in image_names:
        img_path = os.path.join(input_folder_path, img_name)
        img = cv.imread(img_path)
        segmentation = get_segmentation(img, img.shape)
        output_path = os.path.join(output_folder_path, img_name)
        cv.imwrite(output_path, segmentation)