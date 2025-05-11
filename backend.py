import os
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import base64

# Hidden author credentials (decoded at runtime)
_author = base64.b64decode(b'Q2hhaXRhbnlhIEppbmRhbA==').decode()
_email = base64.b64decode(b'amluZGFsY2hhaXRhbnlhQGljbG91ZC5jb20=').decode()

_processor = None
_model = None

def init_model(model_name: str):
    """Initialize the DETR model and processor."""
    global _processor, _model
    if _model is None or _processor is None:
        _processor = DetrImageProcessor.from_pretrained(model_name)
        _model = DetrForObjectDetection.from_pretrained(model_name)
        _model.eval()


def list_images(src_folder: str):
    """Recursively list image file paths under src_folder."""
    image_paths = []
    for root, _, files in os.walk(src_folder):
        for fn in files:
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, fn))
    return image_paths


def detect_objects(image: Image.Image, conf_threshold: float, target_class: str = None):
    """Run DETR inference on a PIL image, filter by class and confidence."""
    inputs = _processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = _model(**inputs)
    detections = _processor.post_process_object_detection(
        outputs,
        threshold=conf_threshold,
        target_sizes=[image.size[::-1]]
    )[0]
    if target_class:
        keep = []
        for score, label, box in zip(detections['scores'], detections['labels'], detections['boxes']):
            if _model.config.id2label[int(label)].lower() == target_class.lower():
                keep.append((score, label, box))
        if keep:
            detections = {
                'scores': torch.stack([s for s,_,_ in keep]),
                'labels': torch.stack([l for _,l,_ in keep]),
                'boxes': torch.stack([b for *_,b in keep], dim=0)
            }
        else:
            detections = {'scores': torch.tensor([]), 'labels': torch.tensor([]), 'boxes': torch.tensor([])}
    return detections


def crop_and_save(image: Image.Image, detections, base_name: str, dst_folder: str):
    """Crop detected boxes from image and save to dst_folder."""
    os.makedirs(dst_folder, exist_ok=True)
    for idx, box in enumerate(detections.get('boxes', [])):
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = image.crop((x1, y1, x2, y2))
        out_name = f"{base_name}_obj_{idx}.jpg"
        crop.save(os.path.join(dst_folder, out_name), format="JPEG")