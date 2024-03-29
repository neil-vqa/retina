import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO, SAM
from ultralytics.engine.results import Results
from lavis.models import load_model_and_preprocess
from pathlib import Path
from typing import Literal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_segmenter_model = YOLO("yolov8x-seg.pt")
sam_segmenter_model = SAM("mobile_sam.pt")
yoloworld_segmenter_model = YOLO("yolov8x-worldv2.pt")


def do_segment(
    image: Path,
    segmenter: Literal["yolo", "sam", "yoloworld"] = "yolo",
    prompt: str = None,
):
    print("processsing image...")
    if segmenter == "yolo":
        results: list[Results] = yolo_segmenter_model.predict(image)
        return results, yolo_segmenter_model
    elif segmenter == "sam":
        results: list[Results] = sam_segmenter_model.predict(image, max_det=15)
        return results, sam_segmenter_model
    elif segmenter == "yoloworld":
        # object detection only; does not perform segmentation
        if prompt:
            yoloworld_segmenter_model.set_classes([prompt])
        results = yoloworld_segmenter_model.predict(image)
        return results, yoloworld_segmenter_model


def rescale_coordinates(original_width, original_height, new_width, new_height, x, y):
    # Calculate the scaling factors for width and height
    sf_w = new_width / original_width
    sf_h = new_height / original_height

    # Scale the coordinates using the scaling factor
    new_x = int(sf_w * x)
    new_y = int(sf_h * y)

    return new_x, new_y


def set_marks(image: Path, detections: list[Results], font_size: int = 20):
    print("setting marks...")
    raw_image = Image.open(image)
    draw = ImageDraw.Draw(raw_image)
    font = ImageFont.load_default(size=font_size)

    for item in detections:
        points = item.masks.xy
        for i, boundary_coords in enumerate(points):
            # using centroid
            # centroid_x = sum(x for x, _ in boundary_coords) / len(boundary_coords)
            # centroid_y = sum(y for _, y in boundary_coords) / len(boundary_coords)
            # centroid = (int(centroid_x), int(centroid_y))

            # using median
            x_coords = boundary_coords[:, 0]
            median_x = np.median(x_coords)
            y_coords = boundary_coords[:, 1]
            median_y = np.median(y_coords)
            centroid = (int(median_x), int(median_y))

            text = str(i + 1)
            text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
            text_background = Image.new(
                "RGB", (text_width + 2, text_height + 2), "black"
            )
            draw = ImageDraw.Draw(text_background)
            draw.text((0, 0), text, fill="white", font=font)

            raw_image.paste(text_background, centroid)

    return raw_image

    # for item in detections:
    #     counter = 0
    #     for mask in item.masks.data:
    #         mask_arr = mask.detach().cpu().numpy() * 255
    #         src_mask = (mask_arr).astype("uint8")

    #         mask_dt = cv2.distanceTransform(src_mask, cv2.DIST_L2, 0)
    #         mask_dt = mask_dt[1:-1, 1:-1]
    #         max_dist = np.max(mask_dt)
    #         coords_y, coords_x = np.where(mask_dt == max_dist)
    #         center = (
    #             coords_x[len(coords_x) // 2] + 2,
    #             coords_y[len(coords_y) // 2] - 6,
    #         )
    #         x, y = center

    #         mask_image = Image.fromarray(mask_arr)
    #         scaled_center = rescale_coordinates(
    #             mask_image.width,
    #             mask_image.height,
    #             raw_image.width,
    #             raw_image.height,
    #             x,
    #             y,
    #         )

    #         text = str(counter + 1)
    #         text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
    #         text_background = Image.new(
    #             "RGB", (text_width + 2, text_height + 2), "black"
    #         )
    #         draw = ImageDraw.Draw(text_background)
    #         draw.text((0, 0), text, fill="white", font=font)

    #         raw_image.paste(text_background, scaled_center)
    #         counter = counter + 1

    # return raw_image


def write_caption(img: Path):
    print("writing image caption...")
    raw_image = Image.open(img).convert("RGB")

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type="large_coco", is_eval=True, device=device
    )
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    caption = model.generate({"image": image})

    return caption[0]


def write_context(img: Path, caption: str):
    """Use GPT-4V"""
    return
