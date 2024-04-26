import os
import albumentations as A
import cv2

transform = A.Compose([
    A.AdvancedBlur(p=1)
    # A.CLAHE(),
    # A.FancyPCA(),
    # A.RandomBrightnessContrast(),
    # A.HorizontalFlip(),
    # A.Normalize()
])

input_image_dir = "dataset/xpm_annotate/images"
input_mask_dir = "dataset/xpm_annotate/masks"
output_image_dir = "dataset/augmented/images"
output_mask_dir = "dataset/augmented/masks"

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

image_filenames = os.listdir(input_image_dir)

for filename in image_filenames:
    image_path = os.path.join(input_image_dir, filename)
    mask_path = os.path.join(input_mask_dir, filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)   

    # -------------------------------------------------------------------------- #
    # --------------------------- APPLY AUGMENTATION --------------------------- #
    # -------------------------------------------------------------------------- #
    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']

    # -------------------------------------------------------------------------- #
    # --------------------------- SAVES AUGMENTATION --------------------------- #
    # -------------------------------------------------------------------------- #
    new_filename = filename.split('.')[0] + "_blur." + filename.split('.')[1]

    output_image_path = os.path.join(output_image_dir, new_filename)
    output_mask_path = os.path.join(output_mask_dir, new_filename)
    cv2.imwrite(output_image_path, transformed_image)
    cv2.imwrite(output_mask_path, transformed_mask)
