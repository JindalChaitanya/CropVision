import os
import sys
import time
from PyQt6 import QtWidgets, QtGui, QtCore
from PIL import Image, ImageDraw
import backend  # assumes backend.py is in the same directory

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DETR Batch Crop GUI")
        self.resize(1200, 800)  # Increased window size
        self.detections = {}

        # Central widget and layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        # Form layout for inputs
        form_layout = QtWidgets.QFormLayout()
        self.model_input = QtWidgets.QLineEdit("facebook/detr-resnet-50")
        self.src_input = QtWidgets.QLineEdit()
        self.src_button = QtWidgets.QPushButton("Browse...")
        self.load_btn = QtWidgets.QPushButton("Load Images")
        self.dst_input = QtWidgets.QLineEdit()
        self.dst_button = QtWidgets.QPushButton("Browse...")
        self.preview_btn = QtWidgets.QPushButton("Preview Detection")
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        self.conf_label = QtWidgets.QLabel("0.50")

        # Source folder layout
        src_layout = QtWidgets.QHBoxLayout()
        src_layout.addWidget(self.src_input)
        src_layout.addWidget(self.src_button)
        src_layout.addWidget(self.load_btn)
        form_layout.addRow("Source Folder:", src_layout)

        # Destination folder layout
        dst_layout = QtWidgets.QHBoxLayout()
        dst_layout.addWidget(self.dst_input)
        dst_layout.addWidget(self.dst_button)
        dst_layout.addWidget(self.preview_btn)
        form_layout.addRow("Destination Folder:", dst_layout)

        # Confidence slider layout
        conf_layout = QtWidgets.QHBoxLayout()
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        form_layout.addRow("Confidence Threshold:", conf_layout)

        # Model input
        form_layout.addRow("Model:", self.model_input)

        main_layout.addLayout(form_layout)

        # Splitter for image list and preview
        splitter = QtWidgets.QSplitter()
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.list_widget.setIconSize(QtCore.QSize(100, 100))
        self.list_widget.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.list_widget.setMovement(QtWidgets.QListView.Movement.Static)
        self.list_widget.setSpacing(10)
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.setWordWrap(True)
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.setWrapping(True)
        self.list_widget.setGridSize(QtCore.QSize(120, 120))
        self.list_widget.setMinimumWidth(400)

        self.preview_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(400, 400)
        self.preview_label.setStyleSheet("border: 1px solid black;")

        splitter.addWidget(self.list_widget)
        splitter.addWidget(self.preview_label)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        # Buttons and progress bar
        btn_layout = QtWidgets.QHBoxLayout()
        self.crop_btn = QtWidgets.QPushButton("Save Croppings")
        self.exit_btn = QtWidgets.QPushButton("Exit")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        btn_layout.addWidget(self.crop_btn)
        btn_layout.addWidget(self.exit_btn)
        btn_layout.addWidget(self.progress_bar)
        main_layout.addLayout(btn_layout)

        # Connect signals
        self.src_button.clicked.connect(self._choose_src)
        self.dst_button.clicked.connect(self._choose_dst)
        self.load_btn.clicked.connect(self.load_images)
        self.conf_slider.valueChanged.connect(self._update_conf_label)
        self.list_widget.currentItemChanged.connect(self.preview_current_image)
        self.list_widget.itemDoubleClicked.connect(self.open_full_image)
        self.preview_label.mouseDoubleClickEvent = self.open_full_preview
        self.preview_btn.clicked.connect(self.preview_current_image)
        self.crop_btn.clicked.connect(self.crop_all)
        self.exit_btn.clicked.connect(self.close)

    def _choose_src(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Source Folder")
        if path:
            self.src_input.setText(path)

    def _choose_dst(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Destination Folder")
        if path:
            self.dst_input.setText(path)

    def _update_conf_label(self, value):
        conf = value / 100.0
        self.conf_label.setText(f"{conf:.2f}")

    def load_images(self):
        src = self.src_input.text()
        if not QtCore.QDir(src).exists():
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid source folder.")
            return
        self.list_widget.clear()
        paths = backend.list_images(src)
        for p in paths:
            item = QtWidgets.QListWidgetItem()
            pixmap = QtGui.QPixmap(p).scaled(100, 100, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
            item.setIcon(QtGui.QIcon(pixmap))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, p)
            self.list_widget.addItem(item)

    def preview_current_image(self):
        item = self.list_widget.currentItem()
        if not item:
            return
        path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        image = Image.open(path).convert("RGB")
        model_name = self.model_input.text().strip()
        backend.init_model(model_name)
        conf = self.conf_slider.value() / 100.0
        dets = backend.detect_objects(image, conf)
        self.detections[path] = dets
        draw = ImageDraw.Draw(image)
        for box in dets['boxes']:
            x1, y1, x2, y2 = map(int, box.tolist())
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        qimg = QtGui.QImage(image.tobytes(), image.width, image.height, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.preview_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.preview_label.setPixmap(pix)

    def open_full_image(self, item):
        path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(os.path.basename(path))
        layout = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        pixmap = QtGui.QPixmap(path)
        label.setPixmap(pixmap.scaled(800, 600, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        layout.addWidget(label)
        dialog.exec()

    def open_full_preview(self, event):
        pixmap = self.preview_label.pixmap()
        if pixmap is None:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Detection Preview")
        layout = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setPixmap(pixmap.scaled(800, 600, QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        layout.addWidget(label)
        dialog.exec()

    def crop_all(self):
        dst = self.dst_input.text()
        if not QtCore.QDir(dst).exists():
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid destination folder.")
            return
        total = len(self.detections)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(0)
        for i, (path, dets) in enumerate(list(self.detections.items()), 1):
            img = Image.open(path).convert("RGB")
            base = os.path.splitext(os.path.basename(path))[0]
            backend.crop_and_save(img, dets, base, dst)
            self.progress_bar.setValue(i)
            QtWidgets.QApplication.processEvents()
            time.sleep(2)  # Simulate heavy computation
        QtWidgets.QMessageBox.information(self, "Done", "Cropped images saved.")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
