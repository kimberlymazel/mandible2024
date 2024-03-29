import cv2
import os

def apply_clahe(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if img is not None:
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
                clahe_img = clahe.apply(img)

                output_filename = os.path.splitext(filename)[0] + os.path.splitext(filename)[1]
                cv2.imwrite(os.path.join(output_dir, output_filename), clahe_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

if __name__ == "__main__":
    input_dir = "../xray_panoramic_mandible/images"
    output_dir = "../xray_panoramic_mandible/images_CLAHE"
    apply_clahe(input_dir, output_dir)
