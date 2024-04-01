import vision
from pathlib import Path

output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)


def go():
    segmenter = "yolo"
    name = "couple"
    ext = "jpg"
    img = f"./sample/{name}.{ext}"
    prompt = None
    output_img_filename = f"./output/{name}-{segmenter}"

    retina = vision.Retina(img, segmenter)
    som = retina.generate_som_image()
    som.save(f"{output_img_filename}-som.{ext}")
    print(f"image saved: {output_img_filename}")

    # caption = retina.write_caption()
    # print(caption)

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


if __name__ == "__main__":
    go()
