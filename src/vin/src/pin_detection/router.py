import numpy as np
from fastapi import APIRouter, Request
from pin_detection import service
from templates import templates
import skimage
from utils import (
    get_raw_image,
    imshow,
    plot_histogram,
    get_mask_image,
    get_mask_region,
    measure_length,
)
from PIL import Image
import cv2 as cv
from measure_order import imgs
import pandas as pd

router = APIRouter(prefix="/pin-detection", tags=["Pin Detection"])


@router.get("/1")
async def detection_1(id: str, request: Request):
    raw = get_raw_image(id, gray=True)
    raw_histogram = await service.histogram(raw)

    mask = get_mask_image(id, gray=True)

    raw = get_mask_region(raw, mask)

    clahe = await service.clahe(raw)
    clahe_histogram = await service.histogram(clahe)

    scharr = await service.scharr_edge_detection(clahe)
    scharr_histogram = await service.histogram(scharr)

    meijering_sato = await service.meijering_sato_ridge_filter(clahe)
    meijering_sato_histogram = await service.histogram(meijering_sato)

    combined = skimage.exposure.rescale_intensity(
        scharr + meijering_sato, in_range="image"
    )
    combined_histogram = await service.histogram(combined)

    return templates.TemplateResponse(
        request=request,
        name="pin_detection_1/index.html",
        context={
            "id": id,
            "raw_image": await imshow(raw),
            "raw_histogram": await plot_histogram(raw_histogram),
            "clahe_image": await imshow(clahe),
            "clahe_histogram": await plot_histogram(clahe_histogram),
            "scharr_image": await imshow(scharr),
            "scharr_histogram": await plot_histogram(scharr_histogram),
            "meijering_sato_image": await imshow(meijering_sato),
            "meijering_sato_histogram": await plot_histogram(meijering_sato_histogram),
            "combined_image": await imshow(combined),
            "combined_histogram": await plot_histogram(combined_histogram),
        },
    )


@router.get("/2")
async def detection_2(id: str, request: Request):
    raw = get_raw_image(id, gray=True)
    mask = get_mask_image(id, gray=True)
    raw = get_mask_region(raw, mask)
    clahe = await service.clahe(raw)
    scharr = await service.scharr_edge_detection(clahe)
    meijering_sato = await service.meijering_sato_ridge_filter(clahe)
    combined = skimage.exposure.rescale_intensity(
        scharr + meijering_sato, in_range="image"
    )
    combined = skimage.util.img_as_ubyte(combined)
    thresholding_segmentation_1 = await service.thresholding_segmentation_1(combined)
    thresholding_segmentation_2 = await service.thresholding_segmentation_2(combined)
    thresholding_segmentation_3 = await service.thresholding_segmentation_3(combined)
    pin_mask, left, right = await service.pin_measurement(thresholding_segmentation_3)
    left_pin_height = await measure_length(
        left[0][0], left[1][0], left[0][1], left[1][1]
    )
    right_pin_height = await measure_length(
        right[0][0], right[1][0], right[0][1], right[1][1]
    )

    WHITE = (255, 255, 255)
    ORANGE = (0, 100, 255)

    pin_mask = cv.cvtColor(pin_mask, cv.COLOR_GRAY2RGBA)
    raw = cv.cvtColor(raw, cv.COLOR_GRAY2RGBA)

    right_mandible_height, left_mandible_height = await service.mandible_height(mask)

    left_height, right_height = (
        left_pin_height / left_mandible_height,
        right_pin_height / right_mandible_height,
    )

    print("left height", left_height)
    print("right height", right_height)

    return templates.TemplateResponse(
        request=request,
        name="pin_detection_2/index.html",
        context={
            "id": id,
            "thresholding_segmentation_1": await imshow(thresholding_segmentation_1),
            "thresholding_segmentation_2": await imshow(thresholding_segmentation_2),
            "thresholding_segmentation_3": await imshow(thresholding_segmentation_3),
            "measurement": f"left: {left_pin_height}px, right: {right_pin_height}px",
            "pin_measurement": await imshow(pin_mask),
        },
    )


@router.get("/all")
async def measure_all():
    data = []
    for i, id in enumerate(imgs):
        raw = get_raw_image(id, gray=True)
        mask = get_mask_image(id, gray=True)
        raw = get_mask_region(raw, mask)
        clahe = await service.clahe(raw)
        scharr = await service.scharr_edge_detection(clahe)
        meijering_sato = await service.meijering_sato_ridge_filter(clahe)
        combined = skimage.exposure.rescale_intensity(
            scharr + meijering_sato, in_range="image"
        )
        try:
            raw = get_raw_image(id, gray=True)
            mask = get_mask_image(id, gray=True)
            raw = get_mask_region(raw, mask)
            clahe = await service.clahe(raw)
            scharr = await service.scharr_edge_detection(clahe)
            meijering_sato = await service.meijering_sato_ridge_filter(clahe)
            combined = skimage.exposure.rescale_intensity(
                scharr + meijering_sato, in_range="image"
            )
            thresholding_segmentation_3 = await service.thresholding_segmentation_3(
                combined
            )

            right_mandible_height, left_mandible_height = await service.mandible_height(
                mask
            )

            pin_mask, left, right = await service.pin_measurement(
                thresholding_segmentation_3
            )

            left_pin_height = await measure_length(
                left[0][0], left[1][0], left[0][1], left[1][1]
            )
            right_pin_height = await measure_length(
                right[0][0], right[1][0], right[0][1], right[1][1]
            )

            left_height, right_height = (
                left_mandible_height / left_pin_height,
                right_mandible_height / right_pin_height,
            )

            d = {"id": id, "left": left_height, "right": right_height}
            data.append(d)
            print(i)

        except Exception as e:
            print(id, e)

        df = pd.DataFrame(data)
        df.to_csv("measurements.csv", index=False)
