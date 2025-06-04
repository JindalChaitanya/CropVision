import os

# --- Default Settings ---

DEFAULT_MODEL_NAME = "yolo11x.pt"
DEFAULT_CONF_THRESHOLD = 0.65
DEFAULT_ITEMS_PER_PAGE = 25
DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "yolo_crops")

# --- Supported Image Formats ---
# Used in core/image_utils.py - ensures consistency
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# --- GUI Settings ---
WINDOW_TITLE = "CropVision v3.1"
WINDOW_ICON = "assets/icon.png"
MIN_WINDOW_WIDTH = 800
MIN_WINDOW_HEIGHT = 600
INITIAL_SPLITTER_RATIO = [1, 2] # Left pane 1/3, Right pane 2/3

# --- Logging ---
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# This is just a fallback/example, it will be populated from the model
DEFAULT_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]