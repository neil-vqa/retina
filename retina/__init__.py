import torch
import numpy as np
import cv2
import os
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from ultralytics.engine.results import Results
from pathlib import Path
from paddleocr import PaddleOCR

load_dotenv()

output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_segmenter_model = YOLO("yolov8x-seg.pt")
ocr_model = PaddleOCR(use_angle_cls=True, lang="en", ocr_version="PP-OCRv4")


class Retina:
    """
    Set of tools for:

    - segmenting an image
    - generating numeric labels on image
    - generating image caption
    - generating image masks
    - generating object crops in image

    """

    def __init__(self, src_image: Path) -> None:
        self.src_image = src_image

    def generate_som_image(self):
        """Generate image with numeric labels useful for Set-of-Mark prompting. Returns a PIL Image."""

        results = self.do_segment()
        som = self.set_marks(results[0], font_size=50)
        return som

    def do_segment(self):
        print("processsing image...")
        results: list[Results] = yolo_segmenter_model.predict(self.src_image)
        return results

    def _rescale_coordinates(
        self, original_width, original_height, new_width, new_height, x, y
    ):
        """
        Rescale point coordinates from mask dimensions to desired image dimensions.

        Args:
            x, y: point coordinates to rescale
        """

        # Calculate the scaling factors for width and height
        sf_w = new_width / original_width
        sf_h = new_height / original_height

        # Scale the coordinates using the scaling factor
        new_x = int(sf_w * x)
        new_y = int(sf_h * y)

        return new_x, new_y

    def _keep_largest_contour(self, binary_mask):
        """
        For mask with multiple parts, keep only the one with largest area for marking.
        """

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

    def set_marks(self, current_image_detections: Results, font_size: int = 20):
        """Takes in a YOLO inference Result. Returns a PIL Image."""

        print("setting marks...")
        raw_image = Image.open(self.src_image)
        raw_image_draw = ImageDraw.Draw(raw_image)
        font = ImageFont.load_default(size=font_size)

        masks = current_image_detections.masks.data
        for i, mask in enumerate(masks):
            mask_arr = mask.detach().cpu().numpy() * 255
            src_mask = (mask_arr).astype("uint8")

            current_mask = self._keep_largest_contour(src_mask)

            mask_dt = cv2.distanceTransform(current_mask, cv2.DIST_L2, 0)
            mask_dt = mask_dt[1:-1, 1:-1]
            max_dist = np.max(mask_dt) * 0.8
            coords_y, coords_x = np.where(mask_dt >= max_dist)

            # calculate a center within image bounds
            height, width = mask_dt.shape
            center_x = coords_x[len(coords_x) // 2] + 2
            center_y = coords_y[len(coords_y) // 2] - 6
            center_x = max(0, min(center_x, width - 1))
            center_y = max(0, min(center_y, height - 1))

            # rescale center point coordinates to image size
            scaled_center = self._rescale_coordinates(
                width,
                height,
                raw_image.width,
                raw_image.height,
                center_x,
                center_y,
            )

            # add numerical label
            text = str(i + 1)
            text_width, text_height = raw_image_draw.textbbox((0, 0), text, font=font)[
                2:
            ]
            text_background = Image.new(
                "RGB", (text_width + 2, text_height + 2), "black"
            )
            draw = ImageDraw.Draw(text_background)
            draw.text((0, 0), text, fill="white", font=font)

            raw_image.paste(text_background, scaled_center)

        return raw_image

    def write_caption(self):
        print("writing image caption...")
        raw_image = Image.open(self.src_image).convert("RGB")
        # image = blip_vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        # caption = caption_model.generate({"image": image})
        caption = "sample"

        return {"caption": caption[0]}

    def get_masked_image(self, current_image_detections: Results, output_filename: str):
        current_image_detections.plot(
            labels=False,
            boxes=False,
            save=True,
            filename=f"{output_filename}",
        )

    def get_image_crops(self, current_image_detections: Results, output_file_id: str):
        current_image_detections.save_crop(save_dir=f"./output/crops/{output_file_id}")


class RetinaOCR:
    def __init__(self, src_image: Path) -> None:
        self.src_image = src_image

    def parse_image(self):
        """Returns a list of tuples: (word, confidence)"""

        result = ocr_model.ocr(self.src_image, cls=True)
        text_detections = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                text_detections.append(line[1])
        return text_detections

    def get_words(self, results: list[tuple]):
        words = []
        for item in results:
            word, _ = item
            words.append(word)
        return words
