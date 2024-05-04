import os
import numpy as np
from PIL import Image

def get_avg_dim(directory):
    total_width = 0
    total_height = 0
    count = 0

    for filename in os.listdir(directory):
        if filename.endswith((".png")):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            width, height = img.size
            total_width += width
            total_height += height
            count += 1

    if count != 0:
        avg_width = total_width / count
        avg_height = total_height / count
        return avg_width, avg_height
    else:
        return None, None

if __name__ == "__main__":
    directory = "dataset/augmented/images" 
    avg_width, avg_height = get_avg_dim(directory)
    
    if avg_width is not None:
        print("Average Width:", avg_width)
        print("Average Height:", avg_height)
    else:
        print("No images found in the directory.")
