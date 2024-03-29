import torch
import numpy as np
import cv2
import pycocotools.mask as mask_util
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


class GenericMask:
    """
    Attribute:
        polygons (list[ndarray]): list[ndarray]: polygons for this mask.
            Each ndarray has format [x, y, x, y, ...]
        mask (ndarray): a binary mask
    """

    def __init__(self, mask_or_polygons, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        m = mask_or_polygons
        if isinstance(m, dict):
            # RLEs
            assert "counts" in m and "size" in m
            if isinstance(m["counts"], list):  # uncompressed RLEs
                h, w = m["size"]
                assert h == height and w == width
                m = mask_util.frPyObjects(m, h, w)
            self._mask = mask_util.decode(m)[:, :]
            return

        if isinstance(m, list):  # list[ndarray]
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return

        if isinstance(m, np.ndarray):  # assumed to be a binary mask
            assert m.shape[1] != 2, m.shape
            assert m.shape == (
                height,
                width,
            ), f"mask shape: {m.shape}, target dims: {height}, {width}"
            self._mask = m.astype("uint8")
            return

        raise ValueError(
            "GenericMask cannot handle object {} of type '{}'".format(m, type(m))
        )

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self.polygons_to_mask(self._polygons)
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
            else:
                self._has_holes = (
                    False  # if original format is polygon, does not have holes
                )
        return self._has_holes

    def mask_to_polygons(self, mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(
            mask
        )  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(
            mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes

    def polygons_to_mask(self, polygons):
        rle = mask_util.frPyObjects(polygons, self.height, self.width)
        rle = mask_util.merge(rle)
        return mask_util.decode(rle)[:, :]

    def area(self):
        return self.mask.sum()

    def bbox(self):
        p = mask_util.frPyObjects(self.polygons, self.height, self.width)
        p = mask_util.merge(p)
        bbox = mask_util.toBbox(p)
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox


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


def keep_largest_contour(binary_mask):
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    max_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    # Create a new mask for the largest contour
    largest_contour_mask = np.zeros_like(binary_mask)
    if largest_contour is not None:
        cv2.drawContours(
            largest_contour_mask, [largest_contour], -1, (255), thickness=cv2.FILLED
        )

    return largest_contour_mask


def is_contour_elongated(contour, ratio_threshold=1.5):
    if len(contour) < 5:
        return False  # Contour is too small for a reliable ellipse fit

    try:
        # Fit an ellipse to the contour
        ellipse = cv2.fitEllipse(contour)
        (center, axes, orientation) = ellipse

        # Calculate the lengths of the major and minor axes
        major_axis_length, minor_axis_length = max(axes), min(axes)

        # Calculate the ratio of the major axis to the minor axis
        axis_ratio = major_axis_length / minor_axis_length

        # Determine if the contour is elongated based on the axis ratio
        return axis_ratio > ratio_threshold
    except cv2.error as e:
        print(f"Error fitting ellipse: {e}")
        return False


def set_marks(image: Path, detections: list[Results], font_size: int = 20):
    print("setting marks...")
    raw_image = Image.open(image)
    draw = ImageDraw.Draw(raw_image)
    font = ImageFont.load_default(size=font_size)

    # using pixel coordinates of masks
    # for item in detections:
    #     points = item.masks.xy
    #     for i, boundary_coords in enumerate(points):
    #         # using centroid
    #         # centroid_x = sum(x for x, _ in boundary_coords) / len(boundary_coords)
    #         # centroid_y = sum(y for _, y in boundary_coords) / len(boundary_coords)
    #         # centroid = (int(centroid_x), int(centroid_y))

    #         # using median
    #         x_coords = boundary_coords[:, 0]
    #         median_x = np.median(x_coords)
    #         y_coords = boundary_coords[:, 1]
    #         median_y = np.median(y_coords)
    #         centroid = (int(median_x), int(median_y))

    #         text = str(i + 1)
    #         text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
    #         text_background = Image.new(
    #             "RGB", (text_width + 2, text_height + 2), "black"
    #         )
    #         draw = ImageDraw.Draw(text_background)
    #         draw.text((0, 0), text, fill="white", font=font)

    #         raw_image.paste(text_background, centroid)

    # return raw_image

    for item in detections:
        masks = item.masks.data
        for i, mask in enumerate(masks):
            mask_arr = mask.detach().cpu().numpy() * 255
            src_mask = (mask_arr).astype("uint8")

            current_mask = keep_largest_contour(src_mask)

            mask_dt = cv2.distanceTransform(current_mask, cv2.DIST_L2, 0)
            mask_dt = mask_dt[1:-1, 1:-1]
            max_dist = np.max(mask_dt)
            coords_y, coords_x = np.where(mask_dt == max_dist)
            center = (
                coords_x[len(coords_x) // 2] + 2,
                coords_y[len(coords_y) // 2] - 6,
            )
            x, y = center

            mask_image = Image.fromarray(mask_arr)
            scaled_center = rescale_coordinates(
                mask_image.width,
                mask_image.height,
                raw_image.width,
                raw_image.height,
                x,
                y,
            )

            text = str(i + 1)
            text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
            text_background = Image.new(
                "RGB", (text_width + 2, text_height + 2), "black"
            )
            draw = ImageDraw.Draw(text_background)
            draw.text((0, 0), text, fill="white", font=font)

            raw_image.paste(text_background, scaled_center)

    return raw_image


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
