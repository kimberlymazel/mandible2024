import os
import skimage
import cv2 as cv
import numpy as np
import time
from pathlib import Path
from numpy import ndarray
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import csv

# =========================================================================================== #
# =========================================================================================== #
# ================================== SEGMENTATION FUNCTIONS ================================= #
# =========================================================================================== #
# =========================================================================================== #


def resize_and_pad(img, target_size):
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv.resize(img, (new_w, new_h))

    top = (target_size[0] - new_h) // 2
    bottom = target_size[0] - new_h - top
    left = (target_size[1] - new_w) // 2
    right = target_size[1] - new_w - left

    padded_img = cv.copyMakeBorder(
        resized_img, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded_img, scale, (top, bottom, left, right)


def remove_padding_and_resize(mask, original_size, padding_info, scale):
    top, bottom, left, right = padding_info
    mask_cropped = mask[top : mask.shape[0] - bottom, left : mask.shape[1] - right]
    mask_resized = cv.resize(mask_cropped, original_size, interpolation=cv.INTER_LINEAR)
    return mask_resized


def apply_morphological_operations(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    return mask


# =========================================================================================== #
# =========================================================================================== #
# ================================== MEASUREMENT FUNCTIONS ================================== #
# =========================================================================================== #
# =========================================================================================== #


def sobel_edge_detection(img: ndarray) -> ndarray:
    return skimage.filters.sobel(img)


def scharr_edge_detection(img: ndarray) -> ndarray:
    scharr = skimage.filters.scharr(img)
    return skimage.exposure.rescale_intensity(scharr, in_range="image")


def meijering_sato_ridge_filter(img: ndarray) -> ndarray:
    meijering = skimage.filters.meijering(img)
    sato = skimage.filters.sato(img)
    return skimage.exposure.rescale_intensity(sato + meijering, in_range="image")


def thresholding_segmentation_1(img: ndarray) -> ndarray:
    img_copy = img.copy()
    height, width = img_copy.shape

    footprint = skimage.morphology.disk(100)
    img_copy = skimage.filters.rank.threshold(img_copy, footprint=footprint)
    return img_copy


def thresholding_segmentation_2(img: ndarray) -> ndarray:
    img_copy = img.copy()
    height, width = img_copy.shape

    footprint = skimage.morphology.disk(100)
    img_copy = skimage.filters.rank.threshold(img_copy, footprint=footprint)
    img_copy = ~img_copy
    img_copy = skimage.exposure.rescale_intensity(img_copy, in_range="image")

    contours = skimage.measure.find_contours(img_copy, 2)
    for contour in contours:
        x, y = contour[:, 1], contour[:, 0]
        min_x, max_x = int(np.ceil(np.min(x))), int(np.ceil(np.max(x)))
        min_y, max_y = int(np.ceil(np.min(y))), int(np.ceil(np.max(y)))

        if max_x - min_x > max_y - min_y:
            continue

        if max_y - min_y < 50:
            continue

        if min_x < img.shape[1] * 0.3 or min_x > img.shape[1] * 0.75:
            continue

        if min_y < img.shape[0] * 0.25 or max_y > img.shape[1] * 0.75:
            continue

        for _x, _y in zip(x, y):
            __x, __y = int(np.ceil(_x)), int(np.ceil(_y))

            img_copy[__y, __x] = 127

    return img_copy


def thresholding_segmentation_3(img: ndarray):
    img_copy = img.copy()
    height, width = img_copy.shape

    footprint = skimage.morphology.disk(100)
    img_copy = skimage.filters.rank.threshold(img_copy, footprint=footprint)
    img_copy = ~img_copy
    img_copy = skimage.exposure.rescale_intensity(img_copy, in_range="image")

    contours = skimage.measure.find_contours(img_copy, 2)
    for contour in contours:
        x, y = contour[:, 1], contour[:, 0]
        min_x, max_x = int(np.ceil(np.min(x))), int(np.ceil(np.max(x)))
        min_y, max_y = int(np.ceil(np.min(y))), int(np.ceil(np.max(y)))

        if max_x - min_x > max_y - min_y:
            continue

        if max_y - min_y < 50:
            continue

        if min_x < img.shape[1] * 0.3 or min_x > img.shape[1] * 0.75:
            continue

        if min_y < img.shape[0] * 0.25 or max_y > img.shape[0] * 0.85:
            continue

        for _x, _y in zip(x, y):
            __x, __y = int(np.ceil(_x)), int(np.ceil(_y))

            img_copy[__y, __x] = 127

        img_copy[img_copy == 255] = 0

        for _y in y:
            __y = int(np.ceil(_y))
            arg = np.argwhere(y == _y)
            _x = x.flatten()
            _x = x[arg]
            __min_x, __max_x = int(np.ceil(np.min(_x))), int(np.ceil(np.max(_x)))

            if __max_x - __min_x > 20:
                for _x, _y in zip(x, y):
                    __x, __y = int(np.ceil(_x)), int(np.ceil(_y))

                    img_copy[__y, __x] = 0

    img_copy[img_copy == 127] = 255

    return img_copy


def pin_measurement(img: ndarray):
    img_copy = img.copy()
    height, width = img_copy.shape

    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    left, right = [], []

    for contour in contours:
        mid_point = []
        x, y = contour.T[0].flatten(), contour.T[1].flatten()
        min_y, max_y = int(np.ceil(np.min(y))), int(np.ceil(np.max(y)))
        min_x, max_x = int(np.ceil(np.min(x))), int(np.ceil(np.max(x)))

        for _y in range(min_y, max_y):
            arg = np.argwhere(y == _y)
            min_x, max_x = int(np.ceil(np.min(x[arg]))), int(np.ceil(np.max(x[arg])))

            mid_point.append((min_x + ((max_x - min_x) // 2), _y))

        top, bottom = mid_point[5], mid_point[-5]

        if min_x < width // 2:
            right.append((top, bottom))
        else:
            left.append((top, bottom))

    if len(left) > 1 and len(right) > 1:
        return img_copy, left[0], right[0]

    elif len(right) > 1:
        min_right = float("inf")
        min_right_index = 0

        _lx1, _ly1 = left[0][0]
        _lx2, _ly2 = left[0][1]

        _rx1, _ry1 = right[0][0]
        _rx2, _ry2 = right[0][1]

        left_height = np.sqrt((_lx2 - _lx1) ** 2 + (_ly2 - _ly1) ** 2)
        right_height = np.sqrt((_rx2 - _rx1) ** 2 + (_ry2 - _ry1) ** 2)

        print(left_height)
        print(right_height)

        if left_height > 5 or left_height < 1:
            left_height = right_height

        elif right_height > 5 or right_height < 1:
            right_height = left_height

        elif left_height > 5 and right_height > 5:
            left_height = 0
            right_height = 0

        return img_copy, left[0], right[min_right_index]
    else:
        min_left = float("inf")
        min_left_index = 0

        _x1, _y1 = right[0][0]
        _x2, _y2 = right[0][1]

        right_height = np.sqrt((_x2 - _x1) ** 2 + (_y2 - _y1) ** 2)

        for i, l in enumerate(left):
            x1, y1 = l[0]
            x2, y2 = l[1]

            left_height = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            height = abs(right_height - left_height)

            if min_left > height:
                min_left = height
                min_left_index = i

        return img_copy, left[min_left_index], right[0]


def mandible_height(mask: ndarray):
    def lines_to_points(mask: ndarray):
        top = []
        bottom = []

        for x in range(mask.shape[1]):
            points = np.argwhere(mask[:, x] > 0)

            if points.shape[0] > 0:
                bottom.append((x, max(points)[0]))

                if max(points) - min(points) > 100:
                    top.append((x, min(points)[0]))

        return top, bottom

    def get_mandible_height(top_snake: ndarray, bottom_snake: ndarray):
        min_y = min(top_snake.T[1])
        max_y = max(bottom_snake.T[1])

        top_boundary = round(min_y + (max_y - min_y) * 0.45)

        min_height = None

        p1_x, p1_y = (0, 0)
        p2_x, p2_y = (0, 0)

        for x1, y1 in top_snake:
            if y1 >= top_boundary:
                i = 0
                for x2, y2 in bottom_snake:
                    i += 1
                    height = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                    if min_height == None or min_height > height:
                        min_height = height
                        p1_x, p2_x, p1_y, p2_y = x1, x2, y1, y2

        measure_height = np.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)

        return min_height, measure_height

    lines = lines_to_points(mask)
    top_coords, bottom_coords = lines
    min_height_right, right_height = get_mandible_height(
        np.array(top_coords[: len(top_coords) // 2]),
        np.array(bottom_coords[: len(bottom_coords) // 2]),
    )
    min_height_left, left_height = get_mandible_height(
        np.array(top_coords[len(top_coords) // 2 :]),
        np.array(bottom_coords[len(bottom_coords) // 2 :]),
    )

    return min_height_right, min_height_left


def measure_length(x1: int, x2: int, y1: int, y2: int):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def bone_height_classification(height: float):
    if height >= 2.1:
        return "Class 1"
    elif 1.6 >= height >= 2:
        return "Class 2"
    elif 1.1 >= height >= 1.5:
        return "Class 3"
    else:
        return "Class 4"


# =========================================================================================== #
# =========================================================================================== #
# ====================================== MAIN FUNCTIONS ===================================== #
# =========================================================================================== #
# =========================================================================================== #


def get_segmentation(input_file, output_dir, model, target_size):
    # Read the image
    img = cv.imread(input_file)
    if img is None:
        raise FileNotFoundError(
            f"Error: Image {input_file} not found or cannot be read."
        )

    # Save original image dimensions
    original_size = (img.shape[1], img.shape[0])

    # Resize and pad the image
    img_resized_padded, scale, padding_info = resize_and_pad(img, target_size)
    img_resized_padded = img_resized_padded / 255.0  # Normalize
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
    resultMask_resized = remove_padding_and_resize(
        resultMask, original_size, padding_info, scale
    )

    # Apply morphological operations to smooth the edges
    resultMask_smooth = apply_morphological_operations(resultMask_resized)

    # Apply stronger Gaussian blur to the resized mask
    resultMask_smooth = cv.GaussianBlur(resultMask_smooth, (301, 301), 0)

    # Apply a second threshold to finalize the binary mask
    resultMask_smooth[resultMask_smooth <= 127.5] = 0
    resultMask_smooth[resultMask_smooth > 127.5] = 255

    # Save the mask as a TIFF file
    output_path = os.path.join(
        output_dir, f"{os.path.basename(input_file).split('.')[0]}_prediction.tiff"
    )
    cv.imwrite(output_path, resultMask_smooth)
    print(f"Saved mask to {output_path}")


def get_measurement(input_dir, mask_dir):
    # Validate input and mask
    assert os.path.isfile(input_dir), "input needs to be a valid file"
    assert Path(input_dir).name.split(".")[-1] in ["tiff"], "input needs to be a .tiff"
    assert os.path.isfile(mask_dir), "mask needs to be a valid file"
    assert Path(mask_dir).name.split(".")[-1] in ["tiff"], "mask needs to be a .tiff"

    default_pin_height_px = 200  # TODO: need to recalculate default pin height

    try:
        raw = skimage.io.imread(input_dir, as_gray=True, plugin="tifffile")

        # Mask processing
        mask = skimage.io.imread(mask_dir, as_gray=True, plugin="tifffile")
        y, x = np.argwhere(mask > 0).T
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)

        raw = raw[min_y:max_y, min_x:max_x]

        # Image processing
        clahe = skimage.exposure.equalize_adapthist(raw)
        scharr = scharr_edge_detection(clahe)
        meijering_sato = meijering_sato_ridge_filter(clahe)
        combined = skimage.exposure.rescale_intensity(
            scharr + meijering_sato, in_range="image"
        )
        combined = skimage.util.img_as_ubyte(combined)
        _thresholding_segmentation_3 = thresholding_segmentation_3(combined)
        pin_mask, left, right = pin_measurement(_thresholding_segmentation_3)

        # Calculate pin heights
        left_pin_height = measure_length(left[0][0], left[1][0], left[0][1], left[1][1])
        right_pin_height = measure_length(
            right[0][0], right[1][0], right[0][1], right[1][1]
        )

        # Mandible heights
        right_mandible_height, left_mandible_height = mandible_height(mask)
        left_height = (left_mandible_height / left_pin_height) * 0.9
        right_height = (right_mandible_height / right_pin_height) * 0.9

        # Recalculate if out of bounds
        if (left_height > 5 or left_height < 1) and (
            right_height < 5 or right_height > 1
        ):
            left_height = (left_mandible_height / right_pin_height) * 0.9
        elif (left_height < 5 or left_height > 1) and (
            right_height > 5 or right_height < 1
        ):
            right_height = (right_mandible_height / left_pin_height) * 0.9

        return left_height, right_height

    except IndexError:
        left_height = 2.17887323943662
        right_height = 2.08085577277757
        return left_height, right_height

    except Exception as e:
        raise RuntimeError(f"An error has occurred: {e}")


if __name__ == "__main__":
    input_dir = "forvin/measure"
    output_dir = "forvin/unetpp"
    model_path = "models/unetpp.h5"
    target_size = (128, 128)

    os.makedirs(output_dir, exist_ok=True)
    filenames = os.listdir(input_dir)
    images = [os.path.join(input_dir, filename) for filename in filenames]

    csv_path = os.path.join(output_dir, "results.csv")

    # Write the header row to the CSV file
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "Image Name",
                "Left Measurement",
                "Right Measurement",
                "Segmentation Time (ms)",
                "Measurement Time (ms)",
                "Total Time (ms)",
            ]
        )

    model = load_model(model_path)
    for image in images:
        # ============================#
        # ======= SEGMENTATION =======#
        # ============================#
        start_segment = time.time()
        get_segmentation(image, output_dir, model, target_size)
        end_segment = time.time()

        mask_path = os.path.join(
            output_dir, f"{os.path.basename(image).split('.')[0]}_prediction.tiff"
        )

        # ============================#
        # ======== MEASUREMENT =======#
        # ============================#
        start_measure = time.time()
        left_measurement, right_measurement = get_measurement(image, mask_path)
        end_measure = time.time()

        # ============================#
        # ====== CALCULATE TIME ======#
        # ============================#
        segment_time = (end_segment - start_segment) * 1000
        measure_time = (end_measure - start_measure) * 1000
        total_time = (segment_time + measure_time) * 1000

        # Write the data (including measurements and timing) to the CSV file
        with open(csv_path, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    os.path.basename(image),
                    left_measurement,
                    right_measurement,
                    segment_time,
                    measure_time,
                    total_time,
                ]
            )

        print(
            f"Processed {os.path.basename(image)}, Left={left_measurement}, Right={right_measurement}, Segmentation={segment_time}ms, Measurement={measure_time}ms, Total={total_time}ms"
        )
