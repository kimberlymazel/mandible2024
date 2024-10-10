import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

def resize_and_pad(img, target_size):
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h))

    top = (target_size[0] - new_h) // 2
    bottom = target_size[0] - new_h - top
    left = (target_size[1] - new_w) // 2
    right = target_size[1] - new_w - left

    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_img, scale, (top, bottom, left, right)

def remove_padding_and_resize(mask, original_size, padding_info, scale):
    top, bottom, left, right = padding_info
    mask_cropped = mask[top:mask.shape[0]-bottom, left:mask.shape[1]-right]
    mask_resized = cv2.resize(mask_cropped, original_size, interpolation=cv2.INTER_LINEAR)
    return mask_resized

def apply_morphological_operations(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def process_images(input_dir, output_dir, model_path, target_size):
    # Load the model
    model = load_model(model_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List of image paths
    filenames = os.listdir(input_dir)
    image_paths = [os.path.join(input_dir, filename) for filename in filenames]

    # Process each image
    for image_path in image_paths:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Image {image_path} not found or cannot be read.")
            continue

        original_size = (img.shape[1], img.shape[0])

        # Resize and pad the image
        img_resized_padded, scale, padding_info = resize_and_pad(img, target_size)
        img_resized_padded = img_resized_padded / 255.0
        img_resized_padded = np.expand_dims(img_resized_padded, axis=0)

        # Predict the mask
        pred = model.predict(img_resized_padded)
        resultMask = pred[0]

        # Apply threshold
        resultMask[resultMask <= 0.5] = 0
        resultMask[resultMask > 0.5] = 255

        # Convert the mask to uint8 type before resizing
        resultMask = resultMask.astype(np.uint8)

        # Remove padding and resize the mask to the original image dimensions
        resultMask_resized = remove_padding_and_resize(resultMask, original_size, padding_info, scale)

        # Apply morphological operations to smooth the edges
        resultMask_smooth = apply_morphological_operations(resultMask_resized)

        # Apply stronger Gaussian blur to the resized mask
        resultMask_smooth = cv2.GaussianBlur(resultMask_smooth, (301, 301), 0)

        resultMask_smooth[resultMask_smooth <= 127.5] = 0
        resultMask_smooth[resultMask_smooth > 127.5] = 255

        # Save the mask as a TIFF file
        output_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_prediction.tiff")
        cv2.imwrite(output_path, resultMask_smooth)
        print(f"Saved mask to {output_path}")

if __name__ == "__main__":
    input_dir = '../../forvin/measure'  # Path to the input image directory
    output_dir = '../../forvin/predicted'  # Path to the output directory for predicted masks
    model_path = "../../models/unetpp_model.h5"  # Path to the trained model file
    target_size = (128, 128) # Target size of the image

    # Process images
    process_images(input_dir, output_dir, model_path, target_size)
