import os
import cv2

def resize_and_pad(image, width, height):
    # Calculate the new dimensions to fit within the target width
    h, w = image.shape[:2]
    aspect_ratio = w / h
    new_w = width
    new_h = int(width / aspect_ratio)
    
    # Resize the image with preserved aspect ratio
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Calculate padding to add to the height
    pad_h = height - new_h
    top = pad_h // 2
    bottom = pad_h - top
    
    if len(resized_image.shape) == 3:  
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:  
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
    
    return padded_image

def process_images(source_dir, dest_dir, width, height):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for filename in os.listdir(source_dir):
        if filename.endswith(".tiff") or filename.endswith(".png"):  # Add other extensions if needed
            image_path = os.path.join(source_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Skipping {filename} as it is not a valid image.")
                continue

            padded_image = resize_and_pad(image, width, height)
            dest_path = os.path.join(dest_dir, filename)
            cv2.imwrite(dest_path, padded_image)
            print(f"Processed and saved {filename} to {dest_dir}")

# Example usage
source_directory = "convert/original"
destination_directory = "convert/padded"
target_width = 1000  # Change as needed
target_height = 1000  # Change as needed

process_images(source_directory, destination_directory, target_width, target_height)
