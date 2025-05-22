# backend.py
import os
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

# Globals
_processor = None
_model = None
_device = None


def init_model(model_name: str = "facebook/detr-resnet-50"):
    """Load DETR model and processor onto available device."""
    global _processor, _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _processor = DetrImageProcessor.from_pretrained(model_name)
        _model = DetrForObjectDetection.from_pretrained(model_name).to(_device)
        _model.eval()
    return _processor, _model, _device


def list_images(src_dir: str, exts=None):
    """Recursively list image files in src_dir."""
    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    paths = []
    for root, _, files in os.walk(src_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                paths.append(os.path.join(root, fn))
    return sorted(paths)


def get_labels(model_name: str = "facebook/detr-resnet-50"):
    """Return class names from DETR model."""
    _, model, _ = init_model(model_name)
    return [model.config.id2label[i] for i in range(len(model.config.id2label))]


def detect_objects(
    image_path: str,
    threshold: float = 0.5,
    target_class: str = None,
    model_name: str = "facebook/detr-resnet-50"
):
    """Run DETR inference, filter by confidence and optional class."""
    processor, model, device = init_model(model_name)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    detections = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=[image.size[::-1]]
    )[0]
    # filter by class if needed
    if target_class:
        keep = []
        for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
            if model.config.id2label[int(label)].lower() == target_class.lower():
                keep.append((score, label, box))
        if keep:
            scores, labels, boxes = zip(*keep)
            detections = {
                "scores": torch.stack(list(scores)),
                "labels": list(labels),
                "boxes": torch.stack(list(boxes))
            }
        else:
            detections = {"scores": torch.tensor([]), "labels": [], "boxes": torch.empty((0,4))}
    return detections


def crop_and_save(
    image_path: str,
    detections: dict,
    output_dir: str,
    prefix: str = ""
):
    """Crop detected boxes and save to output_dir with optional prefix."""
    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    saved = []
    for idx, box in enumerate(detections.get("boxes", [])):
        xmin, ymin, xmax, ymax = box.int().tolist()
        crop = image.crop((xmin, ymin, xmax, ymax))
        base = prefix or os.path.splitext(os.path.basename(image_path))[0]
        fn = f"{base}_{idx}.jpg"
        path = os.path.join(output_dir, fn)
        crop.save(path)
        saved.append(path)
    return saved