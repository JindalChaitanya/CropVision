import os
import sys
from PyQt6 import QtWidgets, QtGui, QtCore
from PIL import Image, ImageDraw
import backend  # assumes backend.py in same directory

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DETR Batch Crop GUI")
        self.detections = {}

        # Widgets setup
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        form = QtWidgets.QFormLayout()
        self.model_input = QtWidgets.QLineEdit("facebook/detr-resnet-50")
        self.src_input = QtWidgets.QLineEdit()
        self.src_button = QtWidgets.QPushButton("Browse...")
        self.dst_input = QtWidgets.QLineEdit()
        self.dst_button = QtWidgets.QPushButton("Browse...")
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        self.conf_label = QtWidgets.QLabel("0.50")

        form.addRow("Model:", self.model_input)
        form.addRow("Source Folder:", self._hbox(self.src_input, self.src_button))
        form.addRow("Destination Folder:", self._hbox(self.dst_input, self.dst_button))
        form.addRow("Confidence:", self._hbox(self.conf_slider, self.conf_label))
        layout.addLayout(form)

        splitter = QtWidgets.QSplitter()
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setIconSize(QtCore.QSize(100, 100))
        self.preview_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        splitter.addWidget(self.list_widget)
        splitter.addWidget(self.preview_label)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        btn_layout = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Load Images")
        self.preview_btn = QtWidgets.QPushButton("Preview Detection")
        self.crop_btn = QtWidgets.QPushButton("Crop & Save")
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.preview_btn)
        btn_layout.addWidget(self.crop_btn)
        layout.addLayout(btn_layout)

        # Signals
        self.src_button.clicked.connect(self._choose_src)
        self.dst_button.clicked.connect(self._choose_dst)
        self.load_btn.clicked.connect(self.load_images)
        self.conf_slider.valueChanged.connect(self._update_conf_label)
        self.list_widget.currentItemChanged.connect(self.preview_current_image)
        self.list_widget.itemDoubleClicked.connect(self.open_full_image)
        self.preview_btn.clicked.connect(self.preview_current_image)
        self.crop_btn.clicked.connect(self.crop_all)

    def _hbox(self, *widgets):
        box = QtWidgets.QHBoxLayout()
        for w in widgets:
            box.addWidget(w)
        container = QtWidgets.QWidget()
        container.setLayout(box)
        return container

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

    def crop_all(self):
        dst = self.dst_input.text()
        if not QtCore.QDir(dst).exists():
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid destination folder.")
            return
        for path, dets in self.detections.items():
            img = Image.open(path).convert("RGB")
            base = os.path.splitext(os.path.basename(path))[0]
            backend.crop_and_save(img, dets, base, dst)
        QtWidgets.QMessageBox.information(self, "Done", "Cropped images saved.")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())