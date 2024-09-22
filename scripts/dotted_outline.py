import os
import cv2 as cv
import numpy as np

def draw_dashed_contours(mask: np.array, dash_length=5, gap_length=0, line_thickness=8):
    # Ensure the mask is in uint8 format
    mask = mask.astype(np.uint8)

    # Find contours in the white mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create an empty image to draw the dashed outline (3 channels for RGB)
    dashed_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Draw dashed contours
    for contour in contours:
        # Iterate over the contour points
        for i in range(0, len(contour), dash_length + gap_length):
            # Draw a short line segment (dash)
            start_idx = i
            end_idx = min(i + dash_length, len(contour) - 1)
            if end_idx > start_idx:
                cv.line(dashed_image, tuple(contour[start_idx][0]), tuple(contour[end_idx][0]), (255, 255, 255), line_thickness)

    return dashed_image

if __name__ == "__main__":
    input_folder_path = os.path.join(os.getcwd(), "convert", "pred")  # Path to the folder with white masks
    output_folder_path = os.path.join(os.getcwd(), "convert", "dots")  # Path to save dashed masks
    
    # Ensure the output folder exists
    os.makedirs(output_folder_path, exist_ok=True)

    # Get the list of mask images
    image_names = list(filter(lambda img: img.split(".")[-1] != "xlsx", os.listdir(input_folder_path)))

    for img_name in image_names:
        img_path = os.path.join(input_folder_path, img_name)
        mask = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # Read the mask as a grayscale image
        
        # Check if the mask was successfully read
        if mask is None:
            print(f"Failed to read {img_path}. Skipping...")
            continue
        
        # Create a dashed outline from the mask
        dashed_outline = draw_dashed_contours(mask)
        
        # Save the dashed outline
        output_path = os.path.join(output_folder_path, img_name)
        cv.imwrite(output_path, dashed_outline)
    
    print("All masks processed with dashed outlines.")