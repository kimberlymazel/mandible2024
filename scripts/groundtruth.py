import os
import cv2 as cv
import numpy as np

def apply_mask_to_image(image: np.array, mask: np.array):
    mask = mask // 255
    segmented_image = cv.bitwise_and(image, image, mask=mask)
    return segmented_image

if __name__ == "__main__":
    mask_folder_path = os.path.join(os.getcwd(), "convert", "masks")  # Folder with white masks (PNG)
    image_folder_path = os.path.join(os.getcwd(), "convert", "original")  # Folder with original images (TIFF)
    output_folder_path = os.path.join(os.getcwd(), "convert", "roi")  # Folder to save segmented images
    
    # Ensure the output folder exists
    os.makedirs(output_folder_path, exist_ok=True)

    # Get the list of mask images (PNG files)
    mask_names = [f for f in os.listdir(mask_folder_path) if f.endswith(".png")]

    # Process each mask
    for mask_name in mask_names:
        # Derive the corresponding image name by replacing .png with .tiff
        base_name = mask_name.rsplit(".", 1)[0]
        image_name = f"{base_name}.tiff"

        mask_path = os.path.join(mask_folder_path, mask_name)
        image_path = os.path.join(image_folder_path, image_name)

        print(f"Attempting to read mask: {mask_path}")
        print(f"Attempting to read image: {image_path}")

        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)  # Read the mask as grayscale
        image = cv.imread(image_path)  # Read the original image

        # Check if the mask and image were successfully read
        if mask is None:
            print(f"Failed to read mask at {mask_path}. Skipping...")
        if image is None:
            print(f"Failed to read image at {image_path}. Skipping...")
        if mask is None or image is None:
            continue

        # Segment the image using the mask
        segmented_image = apply_mask_to_image(image, mask)

        # Save the segmented image
        output_path = os.path.join(output_folder_path, f"{base_name}.png")  # Save as PNG
        cv.imwrite(output_path, segmented_image)

    print("All images segmented and saved.")
