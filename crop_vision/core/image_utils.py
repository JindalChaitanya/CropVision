import os
from PIL import Image
import logging
from .. import config # Import config from the parent package

log = logging.getLogger(__name__)

def list_images(src_dir):
    """
    Recursively finds supported image files in src_dir.
    Returns a sorted list of full paths.
    """
    image_files = []
    log.info(f"Scanning '{src_dir}' for images...")
    try:
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.lower().endswith(config.SUPPORTED_EXTENSIONS):
                    image_files.append(os.path.join(root, file))
        log.info(f"Found {len(image_files)} images.")
        return sorted(image_files)
    except Exception as e:
        log.error(f"Error scanning directory {src_dir}: {e}", exc_info=True)
        return []


def crop_and_save(image_path, detections, output_dir, prefix):
    """
    Crops each box from detections and writes numbered files.
    Returns the number of successfully saved crops.
    """
    if not detections or not detections['boxes']:
        log.warning(f"No detections provided for '{image_path}', cannot crop.")
        return 0

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        log.error(f"Error opening image {image_path}: {e}", exc_info=True)
        return 0

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for i, box in enumerate(detections['boxes']):
        x1, y1, x2, y2 = map(int, box)

        # Clip coordinates to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.width, x2)
        y2 = min(img.height, y2)

        if x1 >= x2 or y1 >= y2:
            log.warning(f"Skipping invalid (zero size) box {i} for {image_path}")
            continue

        cropped_img = img.crop((x1, y1, x2, y2))
        output_filename = os.path.join(output_dir, f"{prefix}_{i}.jpg") # Always save as JPG for consistency? Or keep original? Let's go with JPG.

        try:
            cropped_img.save(output_filename, "JPEG", quality=95)
            log.debug(f"Saved cropped image: {output_filename}")
            count += 1
        except Exception as e:
            log.error(f"Error saving cropped image {output_filename}: {e}", exc_info=True)

    log.info(f"Saved {count} crops from '{image_path}' to '{output_dir}'.")
    return count