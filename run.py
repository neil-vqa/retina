import vision

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


def go():
    segmenter = "yolo"
    name = "couple"
    ext = "jpg"
    img = f"./sample/{name}.{ext}"
    prompt = None
    output_img_filename = f"./output/{name}-{segmenter}"

    results, model = vision.do_segment(img, segmenter=segmenter, prompt=prompt)

    # masked output
    # results[0].plot(
    #     labels=False,
    #     boxes=False,
    #     save=True,
    #     filename=f"{output_img_filename}-mask.{ext}",
    # )
    # print(f"image saved: {output_img_filename}")

    # for COS
    # results[0].save_crop(
    #     save_dir="./output/crops", file_name=f"{name}-crop-{segmenter}"
    # )

    # set SOM
    som = vision.set_marks(img, results, font_size=50)
    som.save(f"{output_img_filename}-som.{ext}")
    print(f"image saved: {output_img_filename}")

    # captioning
    # caption = vision.write_caption(img)
    # print(caption)


if __name__ == "__main__":
    go()


"""
image caption: a woman holding a kitten in her arms

think step by step to write a description of the image (with visual grounding) using both the image and its caption

===

In the image, there's a close-up view of a person (2) and a kitten (1) indoors. The person appears to be a woman, with her face partially visible as she looks down affectionately at the kitten she is holding in her arms. The kitten, comfortably nestled in the woman's embrace, has a greyish coat with some subtle striping and large, expressive eyes that gaze directly towards the viewer. It seems relaxed and content in the woman's hold.

The woman's attire is casual, wearing a t-shirt with a graphic or text on it, which suggests a homely or relaxed setting. The background is blurred and cluttered, indicating an interior space, possibly a workshop or a home with various items scattered around.

The image has a warm and tender feel, capturing a moment of connection between the woman and the kitten. It's a snapshot that speaks to the companionship and comfort pets provide to their humans.
"""
