import os
import traceback
import logging
from PyQt6.QtCore import QRunnable
from .signals import WorkerSignals
from ..core.detector import Detector
from ..core import image_utils

log = logging.getLogger(__name__)

class GenericRunnable(QRunnable):
    """
    Generic QRunnable for executing a function in a thread.
    """
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        log.debug(f"Starting worker for function: {self.fn.__name__}")
        try:
            result_data = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(result_data)
        except Exception as e:
            log.error(f"Error in worker {self.fn.__name__}: {e}", exc_info=True)
            self.signals.error.emit(f"{type(e).__name__}: {str(e)}")
        finally:
            log.debug(f"Finished worker for function: {self.fn.__name__}")
            self.signals.finished.emit()


class BatchProcessingRunnable(QRunnable):
    """
    Specialized QRunnable for batch detection and cropping.
    Emits progress signals.
    """
    def __init__(self, detector: Detector, image_paths: list, threshold: float, class_filter: str, output_dir: str):
        super().__init__()
        self.detector = detector
        self.image_paths = image_paths
        self.threshold = threshold
        self.class_filter = class_filter
        self.output_dir = output_dir
        self.signals = WorkerSignals()
        self.is_cancelled = False

    def run(self):
        log.info(f"Starting batch processing for {len(self.image_paths)} images.")
        total_saved_crops = 0
        total_images = len(self.image_paths)

        if not self.detector.is_loaded():
            self.signals.error.emit("Model is not loaded for batch processing.")
            self.signals.finished.emit()
            return

        for i, img_path in enumerate(self.image_paths):
            if self.is_cancelled:
                self.signals.message.emit("Operation cancelled.")
                break

            try:
                detections = self.detector.detect_objects(img_path, self.threshold, self.class_filter)
                if detections['boxes']:
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    prefix = f"{base_name}_crop"
                    num_saved = image_utils.crop_and_save(img_path, detections, self.output_dir, prefix)
                    total_saved_crops += num_saved
                    self.signals.batch_item_processed.emit(i, f"Processed {os.path.basename(img_path)} - {num_saved} crops.")
                else:
                     self.signals.batch_item_processed.emit(i, f"Processed {os.path.basename(img_path)} - No crops.")

            except Exception as e:
                log.error(f"Error processing {img_path} in batch: {e}", exc_info=True)
                self.signals.batch_item_processed.emit(i, f"ERROR processing {os.path.basename(img_path)}: {e}")

            # Calculate and emit progress (0-100)
            progress_percent = int(((i + 1) / total_images) * 100)
            self.signals.progress.emit(progress_percent)

        if not self.is_cancelled:
            self.signals.result.emit(f"Batch completed. Total crops saved: {total_saved_crops}")

        self.signals.finished.emit()
        log.info("Batch processing finished.")

    def cancel(self):
        log.warning("Cancellation requested for batch processing.")
        self.is_cancelled = True