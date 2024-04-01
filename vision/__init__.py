import torch
import numpy as np
import cv2
import os
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO, SAM
from ultralytics.engine.results import Results
from lavis.models import load_model_and_preprocess
from pathlib import Path
from typing import Literal
from openai import OpenAI

load_dotenv()

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_segmenter_model = YOLO("yolov8x-seg.pt")
sam_segmenter_model = SAM("mobile_sam.pt")


class Retina:
    def __init__(self, src_image: Path, segmenter: str = "yolo") -> None:
        self.src_image = src_image
        self.segmenter = segmenter

    def generate_som_image(self):
        """Returns a PIL Image."""

        results, _ = self.do_segment(segmenter=self.segmenter)
        som = self.set_marks(results, font_size=50)
        return som

    def do_segment(
        self,
        segmenter: Literal["yolo", "sam"] = "yolo",
    ):
        print("processsing image...")
        if segmenter == "yolo":
            results: list[Results] = yolo_segmenter_model.predict(self.src_image)
            return results, yolo_segmenter_model
        elif segmenter == "sam":
            results: list[Results] = sam_segmenter_model.predict(
                self.src_image, max_det=9
            )
            return results, sam_segmenter_model

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

    def set_marks(self, detections: list[Results], font_size: int = 20):
        print("setting marks...")
        raw_image = Image.open(self.src_image)
        raw_image_draw = ImageDraw.Draw(raw_image)
        font = ImageFont.load_default(size=font_size)

        for current_image_detection in detections:
            masks = current_image_detection.masks.data
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
                text_width, text_height = raw_image_draw.textbbox(
                    (0, 0), text, font=font
                )[2:]
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

        model, vis_processors, _ = load_model_and_preprocess(
            name="blip_caption", model_type="large_coco", is_eval=True, device=device
        )
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        caption = model.generate({"image": image})

        return caption[0]


class Chat:
    def __init__(self) -> None:
        """Conversation with OpenAI GPT-4V or any VLM with OpenAI-compatible API"""
        self.model = "gpt-4-vision-preview"

    def write_context(self, image_path: str, caption: str):
        """Use GPT-4V"""
        system_prompt = "You are a visual assistant, whose primary function is to accurately describe and analyze images, and respond to related questions. When describing an image, make sure to mention all key items, and spatial relationships. If asked a question about the image, make use of the numerical markings on the key items, and utilize your analysis to give a concise and applicable response. Always strive for accuracy in your descriptions and responses, and use clear and simple language to ensure understanding by users of varying backgrounds."
        initial_user_prompt = f"Image caption: {caption}\n\nThink step by step to write a description of the image (with visual grounding) using both the image and its caption."

        response = openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": initial_user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_path,
                            },
                        },
                    ],
                },
            ],
            max_tokens=1000,
        )

        return response.choices[0].message


def resize_image(image: Path):
    pass
