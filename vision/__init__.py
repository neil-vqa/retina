import torch
from PIL import Image
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


def write_caption(img: Path):
    raw_image = Image.open(img).convert("RGB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type="large_coco", is_eval=True, device=device
    )
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    caption = model.generate({"image": image})

    return caption[0]


def write_context(img: Path, caption: str):
    """Use GPT-4V"""
    return
