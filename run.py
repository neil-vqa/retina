import vision
from pathlib import Path

output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)


def go():
    segmenter = "yolo"
    name = "coffee-shop"
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
