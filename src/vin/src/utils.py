import os
import cv2 as cv
import numpy as np
from numpy import ndarray
import skimage
from skimage.io import imread
from skimage.util import img_as_ubyte
import plotly.express as px


def image_path(id: str, mask: bool = False):
    if mask:
        return os.path.join(os.getcwd(), "forvin", "predicted", f"{id}_prediction.tiff")
    return os.path.join(os.getcwd(), "forvin", "measure", f"{id}.tiff")


def get_raw_image(id: str, gray: bool):
    raw = imread(image_path(id), as_gray=gray, plugin="tifffile")
    return img_as_ubyte(raw)


def get_mask_image(id: str, gray: bool):
    mask = imread(image_path(id, mask=True), as_gray=gray, plugin="tifffile")
    return img_as_ubyte(mask)


def get_mask_region(img: ndarray, mask: ndarray):
    y, x = np.argwhere(mask > 0).T
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    return img[min_y:max_y, min_x:max_x]


async def histogram(img: ndarray, nbins=256, source_range="image", normalize=False):
    histogram = skimage.exposure.histogram(
        img, nbins=nbins, source_range=source_range, normalize=normalize
    )

    return histogram


async def imshow(img: ndarray, gray=True):
    if gray:
        return px.imshow(img, color_continuous_scale="gray").to_html(full_html=False)
    return px.imshow(img).to_html(full_html=False)


async def plot_histogram(histogram: ndarray):
    return px.bar(x=histogram[1], y=histogram[0]).to_html(full_html=False)


async def measure_length(x1: int, x2: int, y1: int, y2: int):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
