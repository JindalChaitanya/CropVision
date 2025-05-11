import os
import sys
import time
from PyQt6 import QtWidgets, QtGui, QtCore
from PIL import Image, ImageDraw
import backend

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CropVision")
        self.resize(1400, 900)
        self.detections = {}
        self.image_paths = []

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        # Form layout
        form = QtWidgets.QFormLayout()
        self.model_input = QtWidgets.QLineEdit("facebook/detr-resnet-50")
        self.src_input = QtWidgets.QLineEdit()
        self.src_btn = QtWidgets.QPushButton("Browse Src")
        self.load_btn = QtWidgets.QPushButton("Load Images")
        self.dst_input = QtWidgets.QLineEdit()
        self.dst_btn = QtWidgets.QPushButton("Browse Dst")
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        self.conf_label = QtWidgets.QLabel("0.50")
        self.class_combo = QtWidgets.QComboBox()
        self.class_combo.setEditable(True)

        # Source row
        src_layout = QtWidgets.QHBoxLayout()
        src_layout.addWidget(self.src_input)
        src_layout.addWidget(self.src_btn)
        src_layout.addWidget(self.load_btn)
        form.addRow("Source:", src_layout)

        # Destination row
        dst_layout = QtWidgets.QHBoxLayout()
        dst_layout.addWidget(self.dst_input)
        dst_layout.addWidget(self.dst_btn)
        form.addRow("Destination:", dst_layout)

        # Class row
        class_layout = QtWidgets.QHBoxLayout()
        class_layout.addWidget(self.class_combo)
        form.addRow("Class:", class_layout)

        # Confidence row
        conf_layout = QtWidgets.QHBoxLayout()
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        form.addRow("Confidence:", conf_layout)

        form.addRow("Model:", self.model_input)
        main_layout.addLayout(form)

        # Splitter for thumbnails and preview, equal stretch
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.thumb_list = QtWidgets.QListWidget()
        self.thumb_list.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.thumb_list.setIconSize(QtCore.QSize(120, 120))
        self.thumb_list.setGridSize(QtCore.QSize(140, 140))
        self.thumb_list.setSpacing(10)
        self.thumb_list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.thumb_list.setMovement(QtWidgets.QListView.Movement.Static)
        self.thumb_list.setUniformItemSizes(True)
        splitter.addWidget(self.thumb_list)

        self.preview_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("border:1px solid gray;")
        splitter.addWidget(self.preview_label)

        # Make both panes equally stretch
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter, stretch=1)

        # Buttons and progress
        btns = QtWidgets.QHBoxLayout()
        self.preview_btn = QtWidgets.QPushButton("Preview Detection")
        self.crop_btn = QtWidgets.QPushButton("Save Crop")
        self.save_all_btn = QtWidgets.QPushButton("Save All Crops")
        self.exit_btn = QtWidgets.QPushButton("Exit")
        self.progress = QtWidgets.QProgressBar()
        btns.addWidget(self.preview_btn)
        btns.addWidget(self.crop_btn)
        btns.addWidget(self.save_all_btn)
        btns.addWidget(self.exit_btn)
        btns.addWidget(self.progress)
        main_layout.addLayout(btns)

        # Connections
        self.src_btn.clicked.connect(self.choose_src)
        self.dst_btn.clicked.connect(self.choose_dst)
        self.load_btn.clicked.connect(self.load_images)
        self.conf_slider.valueChanged.connect(self.update_conf)
        self.thumb_list.currentItemChanged.connect(self.update_preview)
        self.thumb_list.itemDoubleClicked.connect(self.open_image)
        self.preview_label.mouseDoubleClickEvent = self.open_preview
        self.preview_btn.clicked.connect(self.update_preview)
        self.crop_btn.clicked.connect(self.crop_all)
        self.save_all_btn.clicked.connect(self.save_all_crops)
        self.exit_btn.clicked.connect(self.close)

    def choose_src(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Source")
        if path:
            self.src_input.setText(path)

    def choose_dst(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Destination")
        if path:
            self.dst_input.setText(path)

    def update_conf(self, v):
        c = v/100.0
        self.conf_label.setText(f"{c:.2f}")

    def load_images(self):
        src = self.src_input.text()
        if not QtCore.QDir(src).exists():
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid source folder.")
            return
        self.thumb_list.clear()
        backend.init_model(self.model_input.text())
        self.image_paths = backend.list_images(src)
        # populate class combo now
        self.class_combo.clear()
        self.class_combo.addItems([label.lower() for label in backend._model.config.id2label.values()])
        for p in self.image_paths:
            item = QtWidgets.QListWidgetItem()
            pix = QtGui.QPixmap(p).scaled(120, 120,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation)
            item.setIcon(QtGui.QIcon(pix))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, p)
            self.thumb_list.addItem(item)

    def update_preview(self, _=None):
        item = self.thumb_list.currentItem()
        if not item:
            return
        self._process_and_preview(item.data(QtCore.Qt.ItemDataRole.UserRole))

    def _process_and_preview(self, path):
        img = Image.open(path).convert("RGB")
        backend.init_model(self.model_input.text())
        conf = self.conf_slider.value()/100.0
        cls = self.class_combo.currentText()
        dets = backend.detect_objects(img, conf_threshold=conf, target_class=cls)
        self.detections[path] = dets
        draw = ImageDraw.Draw(img)
        for box in dets.get('boxes', []):
            x1,y1,x2,y2 = map(int, box.tolist())
            draw.rectangle([x1,y1,x2,y2], outline="red", width=2)
        qimg = QtGui.QImage(img.tobytes(), img.width, img.height,
                             QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.preview_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.preview_label.setPixmap(pix)

    def open_image(self, item):
        path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(os.path.basename(path))
        lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl.setPixmap(QtGui.QPixmap(path).scaled(1000,800,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        QtWidgets.QVBoxLayout(dlg).addWidget(lbl)
        dlg.exec()

    def open_preview(self, ev):
        pix = self.preview_label.pixmap()
        if not pix:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Detection Preview")
        lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl.setPixmap(pix.scaled(1000,800,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        QtWidgets.QVBoxLayout(dlg).addWidget(lbl)
        dlg.exec()

    def crop_all(self):
        if not self.detections:
            return
        path, dets = next(iter(self.detections.items()))
        dst = self.dst_input.text()
        if not QtCore.QDir(dst).exists():
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid destination folder.")
            return
        img = Image.open(path).convert("RGB")
        base = os.path.splitext(os.path.basename(path))[0]
        backend.crop_and_save(img, dets, base, dst)
        QtWidgets.QMessageBox.information(self, "Saved", "Current crop saved.")

    def save_all_crops(self):
        dst = self.dst_input.text()
        if not QtCore.QDir(dst).exists():
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid destination folder.")
            return
        total = len(self.image_paths)
        self.progress.setMaximum(total)
        for i, path in enumerate(self.image_paths, 1):
            self._process_and_preview(path)
            img = Image.open(path).convert("RGB")
            base = os.path.splitext(os.path.basename(path))[0]
            dets = self.detections.get(path, {})
            backend.crop_and_save(img, dets, base, dst)
            self.progress.setValue(i)
            QtWidgets.QApplication.processEvents()
        QtWidgets.QMessageBox.information(self, "Done", "All crops saved.")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())