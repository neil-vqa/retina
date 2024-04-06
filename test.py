import retina
from pathlib import Path


def go():
    segmenter = "yolo"
    name = "coffee"
    ext = "jpg"
    img = f"./sample/{name}.{ext}"
    output_img_filename = f"./output/{name}-{segmenter}"

    cur_img = retina.Retina(img)

    # SoM image
    # som = cur_img.generate_som_image()
    # som.save(f"{output_img_filename}-som.{ext}")
    # print(f"image saved: {output_img_filename}")

    # caption
    # res = cur_img.write_caption()
    # print(res["caption"])

    # masked output
    # cur_img.get_masked_image(f"{output_img_filename}-mask.{ext}")

    # for COS
    # cur_img.get_image_crops(name)


if __name__ == "__main__":
    go()
