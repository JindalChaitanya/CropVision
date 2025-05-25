import sys
import os
import logging
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QSlider, QLineEdit, QMessageBox, QSplitter, QProgressDialog, QCompleter,
    QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QThreadPool, pyqtSignal, QSize, QStringListModel, QTimer
)
from PyQt6.QtGui import QPixmap, QIcon, QPainter, QColor, QPen, QGuiApplication, QIcon

from .. import config
from ..core.detector import Detector
from ..core import image_utils
from .workers import GenericRunnable, BatchProcessingRunnable

log = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.WINDOW_TITLE)
        self.setWindowIcon(QIcon(config.WINDOW_ICON))
        
        self.source_dir = ""
        self.dest_dir = config.DEFAULT_OUTPUT_DIR
        self.all_image_files = []
        self.current_page_files = []
        self.current_image_path = None
        self.current_detections = None
        self.current_page = 0
        self.items_per_page = config.DEFAULT_ITEMS_PER_PAGE
        self.current_pixmap = None # Cache the original pixmap
        self.batch_worker = None # To hold reference for cancellation

        self.detector = Detector()
        self.threadpool = QThreadPool()
        log.info(f"Thread pool started with max {self.threadpool.maxThreadCount()} threads.")

        self._set_initial_window_size()
        self._init_ui()
        self.update_button_states()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Pane ---
        left_pane = self._create_left_pane()

        # --- Right Pane ---
        right_pane = self._create_right_pane()

        # --- Splitter ---
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(left_pane)
        self.splitter.addWidget(right_pane)
        self.splitter.setSizes([self.width() // 3, self.width() * 2 // 3]) # 1:2 ratio
        self.splitter.setStretchFactor(0, 1) # Prevent left from shrinking too much
        self.splitter.setStretchFactor(1, 3) # Allow right to expand more

        main_layout.addWidget(self.splitter)

        # --- Resize Timer ---
        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self._handle_resize_finished)


    def _create_left_pane(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Folders
        folder_layout = QHBoxLayout()
        self.source_btn = QPushButton("Source Folder")
        self.source_btn.clicked.connect(self.select_source_dir)
        self.dest_btn = QPushButton("Output Folder")
        self.dest_btn.clicked.connect(self.select_dest_dir)
        folder_layout.addWidget(self.source_btn)
        folder_layout.addWidget(self.dest_btn)
        layout.addLayout(folder_layout)
        self.source_label = QLabel("Source: Not selected")
        self.source_label.setWordWrap(True)
        self.dest_label = QLabel(f"Output: {self.dest_dir}")
        self.dest_label.setWordWrap(True)
        layout.addWidget(self.source_label)
        layout.addWidget(self.dest_label)

        # Model
        model_layout = QHBoxLayout()
        self.model_name_input = QLineEdit(config.DEFAULT_MODEL_NAME)
        self.model_name_input.setPlaceholderText("e.g., yolov8n.pt or path/to/model.pt")
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_name_input)
        model_layout.addWidget(self.load_model_btn)
        layout.addLayout(model_layout)
        self.model_status_label = QLabel("Model: Not loaded")
        layout.addWidget(self.model_status_label)

        # Thumbnails
        self.thumbnail_list_widget = QListWidget()
        self.thumbnail_list_widget.setIconSize(QSize(100, 100))
        self.thumbnail_list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.thumbnail_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.thumbnail_list_widget.setMovement(QListWidget.Movement.Static)
        self.thumbnail_list_widget.currentItemChanged.connect(self.on_thumbnail_selected)
        self.thumbnail_list_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.thumbnail_list_widget)

        # Pagination
        pagination_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.prev_page)
        self.page_label = QLabel("Page 1/1")
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_page)
        pagination_layout.addWidget(self.prev_btn)
        pagination_layout.addWidget(self.page_label, alignment=Qt.AlignmentFlag.AlignCenter)
        pagination_layout.addWidget(self.next_btn)
        layout.addLayout(pagination_layout)

        return widget

    def _create_right_pane(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Preview
        self.image_preview_label = QLabel("Select an image from the left.")
        self.image_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview_label.setMinimumSize(400, 300)
        self.image_preview_label.setStyleSheet("border: 1px solid gray; background-color: #333;")
        self.image_preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.image_preview_label, 1) # Expandable

        # Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        # Threshold
        thresh_layout = QHBoxLayout()
        self.threshold_label = QLabel(f"Confidence Threshold: {config.DEFAULT_CONF_THRESHOLD:.2f}")
        thresh_layout.addWidget(self.threshold_label)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(int(config.DEFAULT_CONF_THRESHOLD * 100))
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        thresh_layout.addWidget(self.threshold_slider)
        controls_layout.addLayout(thresh_layout)

        # Class Filter
        class_filter_layout = QHBoxLayout()
        class_filter_layout.addWidget(QLabel("Class Filter:"))
        self.class_filter_input = QLineEdit()
        self.class_filter_input.setPlaceholderText("e.g., person (leave empty for all)")
        self.class_completer = QCompleter(config.DEFAULT_CLASS_NAMES)
        self.class_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.class_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.class_filter_input.setCompleter(self.class_completer)
        class_filter_layout.addWidget(self.class_filter_input)
        controls_layout.addLayout(class_filter_layout)

        # Actions 1
        actions1_layout = QHBoxLayout()
        self.detect_btn = QPushButton("Detect Objects")
        self.detect_btn.clicked.connect(self.run_detection_on_current)
        actions1_layout.addWidget(self.detect_btn)
        self.delete_btn = QPushButton("Delete Selected Image")
        self.delete_btn.setStyleSheet("background-color: #FF7F7F;")
        self.delete_btn.clicked.connect(self.delete_selected_image)
        actions1_layout.addWidget(self.delete_btn)
        controls_layout.addLayout(actions1_layout)

        # Actions 2
        actions2_layout = QHBoxLayout()
        self.save_crop_btn = QPushButton("Save Current Crop(s)")
        self.save_crop_btn.clicked.connect(self.save_current_image_crops)
        actions2_layout.addWidget(self.save_crop_btn)
        self.save_page_crops_btn = QPushButton("Save Page Crops")
        self.save_page_crops_btn.clicked.connect(self.save_page_images_crops)
        actions2_layout.addWidget(self.save_page_crops_btn)
        self.save_all_crops_btn = QPushButton("Save All Pages Crops")
        self.save_all_crops_btn.clicked.connect(self.save_all_images_crops)
        actions2_layout.addWidget(self.save_all_crops_btn)
        controls_layout.addLayout(actions2_layout)

        layout.addWidget(controls_widget)
        return widget

    def _set_initial_window_size(self):
        primary_screen = QGuiApplication.primaryScreen()
        if not primary_screen:
            self.resize(1000, 750)
            log.warning("Could not get screen info, using default size 1000x750.")
            return

        screen_geom = primary_screen.availableGeometry() # Use available geometry
        s_w, s_h = screen_geom.width(), screen_geom.height()
        log.info(f"Screen available size: {s_w}x{s_h}")

        win_w = int(s_w * 0.7)
        win_h = int(s_h * 0.7)

        # Ensure minimum size and not exceeding 90%
        win_w = max(config.MIN_WINDOW_WIDTH, min(win_w, int(s_w * 0.9)))
        win_h = max(config.MIN_WINDOW_HEIGHT, min(win_h, int(s_h * 0.9)))

        log.info(f"Setting initial window size to: {win_w}x{win_h}")
        self.resize(win_w, win_h)
        # Move to center
        self.move(primary_screen.geometry().center() - self.frameGeometry().center())

    def update_button_states(self):
        has_model = self.detector.is_loaded()
        has_source = bool(self.source_dir and self.all_image_files)
        has_current_image = self.current_image_path is not None
        has_detections = self.current_detections is not None and len(self.current_detections['boxes']) > 0

        self.detect_btn.setEnabled(has_model and has_current_image)
        self.save_crop_btn.setEnabled(has_model and has_current_image and has_detections and bool(self.dest_dir))
        self.save_page_crops_btn.setEnabled(has_model and has_source and bool(self.dest_dir) and len(self.current_page_files) > 0)
        self.save_all_crops_btn.setEnabled(has_model and has_source and bool(self.dest_dir))
        self.delete_btn.setEnabled(has_current_image)

        total_files = len(self.all_image_files)
        if total_files == 0:
            total_pages = 1
            current_p = 0
        else:
            total_pages = (total_files + self.items_per_page - 1) // self.items_per_page
            current_p = self.current_page
        
        self.prev_btn.setEnabled(current_p > 0)
        self.next_btn.setEnabled(current_p < total_pages - 1)
        self.page_label.setText(f"Page {current_p + 1}/{total_pages}")


    def select_source_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Source Directory")
        if dir_path:
            self.source_dir = dir_path
            self.source_label.setText(f"Source: {self.source_dir}")
            log.info(f"Source directory selected: {self.source_dir}")
            self.load_image_files_from_source()
        self.update_button_states()

    def select_dest_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.dest_dir)
        if dir_path:
            self.dest_dir = dir_path
            self.dest_label.setText(f"Output: {self.dest_dir}")
            log.info(f"Destination directory selected: {self.dest_dir}")
        self.update_button_states()

    def load_model(self):
        model_name = self.model_name_input.text().strip()
        if not model_name:
            QMessageBox.warning(self, "Model Name Missing", "Please enter a model name or path.")
            return

        self.model_status_label.setText(f"Loading model: {model_name}...")
        self.load_model_btn.setEnabled(False)

        runnable = GenericRunnable(self.detector.init_model, model_name)
        runnable.signals.result.connect(self.on_model_loaded)
        runnable.signals.error.connect(self.on_model_load_error)
        runnable.signals.finished.connect(lambda: self.load_model_btn.setEnabled(True))
        self.threadpool.start(runnable)

    def on_model_loaded(self, result):
        success, msg = result
        if success:
            model_name = os.path.basename(self.model_name_input.text().strip())
            self.model_status_label.setText(f"Model: {model_name} loaded.")
            QMessageBox.information(self, "Model Loaded", msg)
            class_names = self.detector.get_class_names()
            if class_names:
                completer_model = QStringListModel(class_names)
                self.class_completer.setModel(completer_model)
            else:
                log.warning("Model loaded but no class names found.")
        else:
            self.model_status_label.setText("Model: Load failed.")
            QMessageBox.critical(self, "Model Load Error", msg)
        self.update_button_states()

    def on_model_load_error(self, error_msg):
        self.model_status_label.setText("Model: Load failed.")
        QMessageBox.critical(self, "Model Load Error", f"Failed to load model: {error_msg}")
        self.load_model_btn.setEnabled(True)
        self.update_button_states()

    def load_image_files_from_source(self):
        if not self.source_dir: return
        self.all_image_files = image_utils.list_images(self.source_dir)
        if not self.all_image_files:
            QMessageBox.information(self, "No Images", "No supported image files found in the selected directory.")
        self.current_page = 0
        self.update_thumbnails_for_page()
        self.update_button_states()

    def update_thumbnails_for_page(self):
        self.thumbnail_list_widget.clear()
        self.current_page_files = []
        if not self.all_image_files:
            self.update_button_states()
            return

        start_idx = self.current_page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, len(self.all_image_files))
        self.current_page_files = self.all_image_files[start_idx:end_idx]

        for img_path in self.current_page_files:
            try:
                pixmap = QPixmap(img_path)
                item = QListWidgetItem(QIcon(pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)), os.path.basename(img_path))
                item.setData(Qt.ItemDataRole.UserRole, img_path)
                self.thumbnail_list_widget.addItem(item)
            except Exception as e:
                log.error(f"Error loading thumbnail for {img_path}: {e}")
        
        # Select first item if none selected
        if self.thumbnail_list_widget.count() > 0 and self.thumbnail_list_widget.currentRow() == -1:
            self.thumbnail_list_widget.setCurrentRow(0)

        self.update_button_states()


    def on_thumbnail_selected(self, current_item, _previous_item):
        if current_item:
            self.current_image_path = current_item.data(Qt.ItemDataRole.UserRole)
            self.current_detections = None # Clear detections for new image
            self.load_and_display_image(self.current_image_path)
        else:
            self.current_image_path = None
            self.current_pixmap = None
            self.image_preview_label.setText("No image selected.")
            self.image_preview_label.setPixmap(QPixmap())
        self.update_button_states()

    def load_and_display_image(self, image_path):
        """Loads the image into current_pixmap and calls display."""
        if not image_path:
            self.current_pixmap = None
            self.display_image()
            return

        try:
            self.current_pixmap = QPixmap(image_path)
            if self.current_pixmap.isNull():
                log.error(f"Failed to load image: {image_path}")
                self.image_preview_label.setText(f"Error: Could not load image\n{os.path.basename(image_path)}")
                self.current_pixmap = None
            self.display_image() # Display initially without boxes
        except Exception as e:
            log.error(f"Error loading {image_path}: {e}", exc_info=True)
            self.current_pixmap = None
            self.display_image()


    def display_image(self, detections=None):
        """Displays the current_pixmap, optionally drawing detections."""
        if not self.current_pixmap or self.current_pixmap.isNull():
            self.image_preview_label.clear()
            self.image_preview_label.setText("No image to display.")
            return

        pixmap_to_show = self.current_pixmap.copy() # Work on a copy

        if detections and detections['boxes']:
            painter = QPainter(pixmap_to_show)
            pen = QPen(QColor("red"), max(2, int(pixmap_to_show.width() / 400))) # Scale pen width
            painter.setPen(pen)
            font = painter.font()
            font.setPointSize(max(8, int(pixmap_to_show.height() / 60)))
            painter.setFont(font)

            for i, box in enumerate(detections['boxes']):
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                painter.drawRect(x1, y1, w, h)

                label = detections['labels'][i]
                score = detections['scores'][i]
                text = f"{label}: {score:.2f}"

                metrics = painter.fontMetrics()
                text_width = metrics.horizontalAdvance(text) + 4
                text_height = metrics.height()
                text_x, text_y = x1, y1 - 2

                if text_y < text_height: text_y = y1 + h + text_height

                painter.fillRect(text_x, text_y - text_height, text_width, text_height, QColor(255, 0, 0, 180))
                painter.setPen(QColor("white"))
                painter.drawText(text_x + 2, text_y -1, text)
                painter.setPen(QColor("red")) # Reset pen

            painter.end()

        # Scale and display
        self.image_preview_label.setPixmap(pixmap_to_show.scaled(
            self.image_preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        
        self.current_detections = detections # Store detections


    def update_threshold_label(self, value):
        self.threshold_label.setText(f"Confidence Threshold: {value / 100.0:.2f}")
        # Optionally re-run detection or just re-draw if detections exist
        if self.current_detections:
             self.run_detection_on_current() # Re-run to re-filter

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_thumbnails_for_page()
        self.update_button_states()

    def next_page(self):
        total_pages = (len(self.all_image_files) + self.items_per_page - 1) // self.items_per_page
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.update_thumbnails_for_page()
        self.update_button_states()

    def run_detection_on_current(self):
        if not self.current_image_path or not self.detector.is_loaded():
            QMessageBox.warning(self, "Cannot Detect", "Please select an image and load a model first.")
            return

        threshold = self.threshold_slider.value() / 100.0
        class_filter = self.class_filter_input.text().strip()

        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("Detecting...")

        runnable = GenericRunnable(self.detector.detect_objects, self.current_image_path, threshold, class_filter)
        runnable.signals.result.connect(self.on_detection_complete)
        runnable.signals.error.connect(self.on_task_error)
        runnable.signals.finished.connect(lambda: (
            self.detect_btn.setText("Detect Objects"),
            self.update_button_states()
        ))
        self.threadpool.start(runnable)

    def on_detection_complete(self, detections):
        if not detections or not detections['boxes']:
             QMessageBox.information(self, "Detection Complete", "No objects found matching criteria.")
             self.display_image(None) # Show original image
        else:
             num_found = len(detections['boxes'])
             log.info(f"Detection found {num_found} objects.")
             self.display_image(detections) # Redraw with boxes
        self.update_button_states()


    def save_current_image_crops(self):
        if not self.current_image_path or not self.current_detections or not self.dest_dir:
            QMessageBox.warning(self, "Cannot Save", "No image, detections, or output directory.")
            return
        if not self.current_detections['boxes']:
            QMessageBox.information(self, "No Detections", "No objects to save.")
            return

        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        prefix = f"{base_name}_crop"
        self.save_crop_btn.setEnabled(False)
        self.save_crop_btn.setText("Saving...")

        runnable = GenericRunnable(image_utils.crop_and_save, self.current_image_path, self.current_detections, self.dest_dir, prefix)
        runnable.signals.result.connect(lambda count: QMessageBox.information(self, "Crops Saved", f"Saved {count} cropped image(s) to {self.dest_dir}."))
        runnable.signals.error.connect(self.on_task_error)
        runnable.signals.finished.connect(lambda: (
            self.save_crop_btn.setText("Save Current Crop(s)"),
            self.update_button_states()
        ))
        self.threadpool.start(runnable)


    def _batch_save_crops(self, image_paths, operation_name):
        if not self.detector.is_loaded() or not self.dest_dir or not image_paths:
            QMessageBox.warning(self, "Cannot Save", "Model not loaded, output dir not set, or no images.")
            return

        threshold = self.threshold_slider.value() / 100.0
        class_filter = self.class_filter_input.text().strip()

        self.progress_dialog = QProgressDialog(f"{operation_name}...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(False) # We will close manually
        self.progress_dialog.setAutoReset(True)
        self.progress_dialog.setValue(0)

        self.batch_worker = BatchProcessingRunnable(
            self.detector, image_paths, threshold, class_filter, self.dest_dir
        )

        self.progress_dialog.canceled.connect(self.batch_worker.cancel)
        self.batch_worker.signals.progress.connect(self.progress_dialog.setValue)
        self.batch_worker.signals.message.connect(lambda msg: log.info(f"Batch message: {msg}")) # Or show in status bar
        self.batch_worker.signals.batch_item_processed.connect(lambda i, msg: self.progress_dialog.setLabelText(f"Processing ({i+1}/{len(image_paths)}): {msg}"))
        self.batch_worker.signals.result.connect(lambda result_msg: QMessageBox.information(self, operation_name, result_msg))
        self.batch_worker.signals.error.connect(self.on_task_error)
        self.batch_worker.signals.finished.connect(self._on_batch_finished)

        self.save_page_crops_btn.setEnabled(False)
        self.save_all_crops_btn.setEnabled(False)
        self.progress_dialog.show()
        self.threadpool.start(self.batch_worker)

    def _on_batch_finished(self):
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        self.batch_worker = None
        self.update_button_states()
        log.info("Batch processing GUI cleanup finished.")


    def save_page_images_crops(self):
        self._batch_save_crops(list(self.current_page_files), "Save Page Crops")

    def save_all_images_crops(self):
        self._batch_save_crops(list(self.all_image_files), "Save All Pages Crops")


    def delete_selected_image(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Cannot Delete", "No image selected.")
            return

        reply = QMessageBox.question(self, "Confirm Delete",
                                       f"Are you sure you want to permanently delete\n'{os.path.basename(self.current_image_path)}'?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                path_to_delete = self.current_image_path # Store before clearing
                log.warning(f"User initiated delete for: {path_to_delete}")
                os.remove(path_to_delete)
                QMessageBox.information(self, "Deleted", f"File '{os.path.basename(path_to_delete)}' deleted.")

                # Remove from lists and update UI
                if path_to_delete in self.all_image_files:
                    self.all_image_files.remove(path_to_delete)

                self.current_image_path = None
                self.current_detections = None
                self.current_pixmap = None
                self.image_preview_label.clear()
                self.image_preview_label.setText("Select an image.")

                # Recalculate page and update thumbnails
                total_files = len(self.all_image_files)
                if total_files == 0:
                    self.current_page = 0
                else:
                    total_pages = (total_files + self.items_per_page - 1) // self.items_per_page
                    if self.current_page >= total_pages:
                        self.current_page = max(0, total_pages - 1)
                
                self.update_thumbnails_for_page()

            except Exception as e:
                log.error(f"Could not delete file {path_to_delete}: {e}", exc_info=True)
                QMessageBox.critical(self, "Delete Error", f"Could not delete file: {e}")
        self.update_button_states()

    def on_task_error(self, error_msg):
        QMessageBox.critical(self, "Operation Error", str(error_msg))
        log.error(f"GUI received task error: {error_msg}")
        self.update_button_states() # Ensure buttons reset


    def resizeEvent(self, event):
        """Handle window resizing, start timer for debounced redraw."""
        super().resizeEvent(event)
        self.resize_timer.start(100) # Wait 100ms after last resize event

    def _handle_resize_finished(self):
        """Called by timer; redraws the image preview to fit."""
        if self.current_image_path:
            log.debug("Handling resize - redrawing image.")
            self.display_image(self.current_detections)

    def closeEvent(self, event):
        """Handle window closing: ensure threads are handled."""
        log.info("Close event received. Cleaning up.")
        if self.batch_worker and self.batch_worker.is_cancelled == False:
             reply = QMessageBox.question(self, "Confirm Close",
                                       "A batch process is running. Are you sure you want to exit?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No)
             if reply == QMessageBox.StandardButton.Yes:
                 self.batch_worker.cancel()
                 self.threadpool.waitForDone(3000) # Wait 3s
                 event.accept()
             else:
                 event.ignore()
        else:
            self.threadpool.waitForDone(1000) # Wait 1s
            event.accept()