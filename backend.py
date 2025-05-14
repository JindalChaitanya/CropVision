import os
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

# Globals for processor and model
_processor = None
_model = None
_device = None

def init_model(model_name: str = "facebook/detr-resnet-50"):
    global _processor, _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _processor = DetrImageProcessor.from_pretrained(model_name)
        _model = DetrForObjectDetection.from_pretrained(model_name).to(_device)
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

def get_labels():
    """Return list of class names from the loaded model."""
    _, model, _ = init_model()
    return [model.config.id2label[i] for i in range(len(model.config.id2label))]

def detect_objects(
    image_path: str,
    threshold: float = 0.5,
    target_class: str = None,
):
    """
    Run DETR inference on a single image.
    Returns dict with 'scores', 'labels', 'boxes' (tensor Nx4 in xyxy).
    """
    processor, model, device = init_model()
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to open image {image_path}: {e}")

    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits.softmax(-1)[0]  # (num_queries, num_classes+1)
    boxes = outputs.pred_boxes[0]           # (num_queries, 4)

    # filter out low-confidence & "no-object"
    keep = []
    for score_vec, box in zip(logits, boxes):
        conf, class_idx = torch.max(score_vec[:-1], dim=0)
        if conf >= threshold:
            label = model.config.id2label[class_idx.item()]
            if target_class is None or label == target_class:
                # convert cxcywh [0,1] to xyxy absolute
                w, h = image.size
                cx, cy, bw, bh = box
                xmin = (cx - bw / 2) * w
                ymin = (cy - bh / 2) * h
                xmax = (cx + bw / 2) * w
                ymax = (cy + bh / 2) * h
                keep.append((conf, label, torch.tensor([xmin, ymin, xmax, ymax])))

    if not keep:
        return {"scores": torch.tensor([]), "labels": [], "boxes": torch.empty((0, 4))}

    scores, labels, bboxes = zip(*keep)
    return {
        "scores": torch.stack(list(scores)),
        "labels": list(labels),
        "boxes": torch.stack(list(bboxes)),
    }

def crop_and_save(
    image_path: str,
    detections: dict,
    output_dir: str,
    prefix: str = ""
):
    """
    Crop detected boxes and save to output_dir, filenames prefixed by prefix.
    """
    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    saved = []
    for idx, box in enumerate(detections["boxes"]):
        xmin, ymin, xmax, ymax = box.int().tolist()
        crop = image.crop((xmin, ymin, xmax, ymax))
        fn = f"{prefix}_{idx}.png"
        path = os.path.join(output_dir, fn)
        crop.save(path)
        saved.append(path)
    return saved