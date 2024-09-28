import skimage
import cv2 as cv
import numpy as np
from numpy import ndarray


async def histogram(img: ndarray) -> ndarray:
    return skimage.exposure.histogram(img)


async def sobel_edge_detection(img: ndarray) -> ndarray:
    return skimage.filters.sobel(img)


async def scharr_edge_detection(img: ndarray) -> ndarray:
    scharr = skimage.filters.scharr(img)
    return skimage.exposure.rescale_intensity(scharr, in_range="image")


async def meijering_sato_ridge_filter(img: ndarray) -> ndarray:
    meijering = skimage.filters.meijering(img)
    sato = skimage.filters.sato(img)
    return skimage.exposure.rescale_intensity(sato + meijering, in_range="image")


async def clahe(img: ndarray) -> ndarray:
    return skimage.exposure.equalize_adapthist(img)


async def thresholding_segmentation_1(img: ndarray) -> ndarray:
    img_copy = img.copy()
    height, width = img_copy.shape

    footprint = skimage.morphology.disk(100)
    img_copy = skimage.filters.rank.threshold(img_copy, footprint=footprint)
    return img_copy


async def thresholding_segmentation_2(img: ndarray) -> ndarray:
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


async def thresholding_segmentation_3(img: ndarray):
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


async def pin_measurement(img: ndarray):
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
            right = top, bottom
        else:
            left = top, bottom

    return img_copy, left, right


async def mandible_height(mask: ndarray):
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
