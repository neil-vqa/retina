import vision

# from ultralytics.utils import plotting


def go():
    segmenter = "yolo"
    name = "coffee-shop"
    img = f"./sample/{name}.jpg"
    prompt = None
    operation = "crop"

    # results, model = vision.do_segment(img, segmenter=segmenter, prompt=prompt)

    # results[0].save_crop(
    #     save_dir="./output/crops", file_name=f"{name}-mask-{segmenter}"
    # )
    # image_filename = f"./output/{name}-mask-{segmenter}.jpg"
    # results[0].save(filename=image_filename)
    # results[0].plot(labels=False, boxes=False, save=True, filename=image_filename)
    # print(f"image saved: {image_filename}")
    # print(results[0].boxes.xyxy)

    caption = vision.write_caption(img)
    print(caption)

    # sv workflow
    # detections = sv.Detections.from_ultralytics(results[0])

    # image = cv2.imread(img)
    # mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    # label_annotator = sv.LabelAnnotator()

    # labels = [model.model.names[class_id] for class_id in detections.class_id]

    # annotated_image = mask_annotator.annotate(scene=image, detections=detections)
    # annotated_image = label_annotator.annotate(
    #     scene=annotated_image, detections=detections, labels=labels
    # )

    # with sv.ImageSink(target_dir_path="./output", overwrite=True) as sink:
    #     image_filename = f"{name}-mask-{segmenter}.jpg"
    #     sink.save_image(image=annotated_image, image_name=image_filename)
    #     print(f"image saved: {image_filename}")


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
