import os
import cv2 as cv
import numpy as np

def get_annotation_lines(img: np.array):
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_img, (25, 25, 25), (255, 255, 255))
    return mask

def fill_segmentation(mask: np.array):
    for x in range(mask.shape[1]):
        points = np.argwhere(mask[:, x] > 0)
        
        if points.shape[0] > 0:
            mask[min(points[2:])[0]:max(points)[0] + 1, x] = 255
    
    return mask

if __name__ == "__main__":
    input_folder_path = os.path.join(os.getcwd(), "dataset", "annotations2") # Change folder path
    image_names = list(filter(lambda img: img.split(".")[-1] != "xlsx" and img.split(".")[0][-1] == "2", os.listdir(input_folder_path))) # Change image title filter
    output_folder_path = os.path.join(os.getcwd(), "dataset", "segmentation")
    
    for img_name in image_names:
        img_path = os.path.join(input_folder_path, img_name)
        img = cv.imread(img_path)
        mask = get_annotation_lines(img)
        segmentation = fill_segmentation(mask)
        output_path = os.path.join(output_folder_path, img_name)
        cv.imwrite(output_path, segmentation)
    
    print("All files segmented.")