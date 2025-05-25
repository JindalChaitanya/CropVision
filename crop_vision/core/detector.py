import os
import torch
from ultralytics import YOLO
from PIL import Image
import logging

log = logging.getLogger(__name__)

class Detector:
    """Encapsulates the YOLO model and detection logic."""

    def __init__(self):
        self.model = None
        self.device = None
        self.class_names = []

    def init_model(self, model_name_or_path):
        """
        Loads a YOLO model, moves it to CPU/GPU, and performs a dummy inference.
        Returns (success, message_or_error).
        """
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            log.info(f"Attempting to load model '{model_name_or_path}' on {self.device}...")

            self.model = YOLO(model_name_or_path)
            self.model.to(self.device)

            # Perform a dummy inference
            dummy_img = Image.new('RGB', (64, 64), color='red')
            results = self.model(dummy_img, verbose=False)

            self.class_names = list(results[0].names.values()) if results and results[0].names else []
            log.info(f"Model '{model_name_or_path}' loaded successfully on {self.device}.")
            log.debug(f"Model classes: {self.class_names}")

            return True, f"Model '{model_name_or_path}' loaded on {self.device}."
        except Exception as e:
            self.model = None
            self.device = None
            self.class_names = []
            log.error(f"Error loading model: {e}", exc_info=True)
            return False, str(e)

    def is_loaded(self):
        """Checks if a model is currently loaded."""
        return self.model is not None

    def get_class_names(self):
        """Returns the list of class names from the loaded model."""
        return self.class_names

    def detect_objects(self, image_path, threshold, target_class=None):
        """
        Runs YOLO inference, filters by confidence and optional class.
        Returns {scores, labels, boxes}.
        Raises ValueError if model not loaded or inference error.
        """
        if not self.is_loaded():
            raise ValueError("Model not initialized. Call init_model() first.")

        log.debug(f"Running detection on '{image_path}' with threshold {threshold} and class '{target_class}'")
        try:
            results = self.model(image_path, verbose=False)
        except Exception as e:
            log.error(f"Error during model inference for {image_path}: {e}", exc_info=True)
            raise RuntimeError(f"Model inference failed for {os.path.basename(image_path)}: {e}")

        if not results or len(results) == 0:
            return {'scores': [], 'labels': [], 'boxes': []}

        pred = results[0]
        all_boxes = pred.boxes.xyxy.cpu().numpy()
        all_scores = pred.boxes.conf.cpu().numpy()
        all_class_ids = pred.boxes.cls.cpu().numpy().astype(int)
        model_class_names = pred.names

        filtered_scores = []
        filtered_labels = []
        filtered_boxes = []

        for i in range(len(all_scores)):
            score = all_scores[i]
            if score >= threshold:
                class_id = all_class_ids[i]
                label = model_class_names.get(class_id, f"ID_{class_id}") # Use .get for safety

                if target_class and target_class.strip():
                    if label.lower() != target_class.strip().lower():
                        continue  # Skip if class doesn't match

                filtered_scores.append(score)
                filtered_labels.append(label)
                filtered_boxes.append(all_boxes[i])

        log.debug(f"Found {len(filtered_boxes)} objects matching criteria.")
        return {
            'scores': filtered_scores,
            'labels': filtered_labels,
            'boxes': filtered_boxes
        }