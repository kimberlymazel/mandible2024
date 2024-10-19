import os
import numpy as np
from numpy import ndarray
import cv2 as cv
from pathlib import Path
import asyncio
import skimage
import argparse
from time import time


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

        _x1, _y1 = left[0][0]
        _x2, _y2 = left[0][1]

        left_height = np.sqrt((_x2 - _x1) ** 2 + (_y2 - _y1) ** 2)
        for i, r in enumerate(right):
            x1, y1 = r[0]
            x2, y2 = r[1]

            right_height = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            height = abs(right_height - left_height)
            if min_right > height:
                min_right = height
                min_right_index = i

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input file")
    parser.add_argument("mask", help="Mask file")

    args = parser.parse_args()

    assert os.path.isfile(args.input), "input need to be a valid file"
    assert Path(args.input).name.split(".")[-1] in ["tiff"], "input need to be a .tiff"

    assert os.path.isfile(args.mask), "mask need to be a valid file"
    assert Path(args.mask).name.split(".")[-1] in ["tiff"], "mask need to be a .tiff"

    start = time()
    raw = skimage.io.imread(args.input, as_gray=True, plugin="tifffile")
    mask = skimage.io.imread(args.mask, as_gray=True, plugin="tifffile")

    y, x = np.argwhere(mask > 0).T
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    raw = raw[min_y:max_y, min_x:max_x]

    clahe = skimage.exposure.equalize_adapthist(raw)
    scharr = scharr_edge_detection(clahe)
    meijering_sato = meijering_sato_ridge_filter(clahe)
    combined = skimage.exposure.rescale_intensity(
        scharr + meijering_sato, in_range="image"
    )
    combined = skimage.util.img_as_ubyte(combined)
    _thresholding_segmentation_3 = thresholding_segmentation_3(combined)
    pin_mask, left, right = pin_measurement(_thresholding_segmentation_3)
    left_pin_height = measure_length(left[0][0], left[1][0], left[0][1], left[1][1])
    right_pin_height = measure_length(
        right[0][0], right[1][0], right[0][1], right[1][1]
    )

    right_mandible_height, left_mandible_height = mandible_height(mask)

    left_height, right_height = (
        (left_mandible_height / left_pin_height) * 0.8,
        (right_mandible_height / right_pin_height) * 0.8,
    )

    print("left height", left_height, "cm")
    print("right height", right_height, "cm")

    end = time()
