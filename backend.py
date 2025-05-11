import os
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image

# Initialize model and processor
_processor = None
_model = None

def init_model(model_name: str):
    global _processor, _model
    if _model is None or _processor is None:
        _processor = DetrImageProcessor.from_pretrained(model_name)
        _model = DetrForObjectDetection.from_pretrained(model_name)
        _model.eval()


def list_images(src_folder: str):
    """
    Recursively list image file paths under src_folder.
    """
    image_paths = []
    for root, _, files in os.walk(src_folder):
        for fn in files:
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, fn))
    return image_paths


def detect_objects(image: Image.Image, conf_threshold: float):
    """
    Run DETR inference on a PIL image and return detections above conf_threshold.
    """
    inputs = _processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = _model(**inputs)
    detections = _processor.post_process_object_detection(
        outputs,
        threshold=conf_threshold,
        target_sizes=[image.size[::-1]]
    )[0]
    return detections


def crop_and_save(image: Image.Image, detections, base_name: str, dst_folder: str):
    """
    Crop detected boxes from image and save them to dst_folder.
    """
    os.makedirs(dst_folder, exist_ok=True)
    for idx, box in enumerate(detections['boxes']):
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = image.crop((x1, y1, x2, y2))
        out_name = f"{base_name}_obj_{idx}.jpg"
        crop.save(os.path.join(dst_folder, out_name), format="JPEG")