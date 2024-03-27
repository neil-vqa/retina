# som-blip-cos

Demonstration of combining the concepts of Set-of-Mark (SoM) Prompting, Chain-of-Spot (CoS) approach, with BLIP captions for better visual question answering.

The pipeline starts with an input image. Captioning and generating a marked image prompt (SoM image) will be done simultaneously. BLIP will be used for captioning. SoM image will be generated with the help of YOLOv8-seg or SAM. The text caption and SoM image will then be treated as input prompts to GPT-4V to write a description with visual grounding. The written description will be used as a context for succeeding chat interactions together with the SoM image.

## References

[SoM : Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V](https://som-gpt4v.github.io)

[Chain-of-Spot: Interactive Reasoning Improves Large Vision-language Models](https://sites.google.com/view/chain-of-spot)

[LLaVA-Grounding: Grounded Visual Chat with Large Multimodal Models](https://llava-vl.github.io/llava-grounding)

[Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)

[BLIP captioning](https://github.com/salesforce/LAVIS?tab=readme-ov-file#image-captioning)