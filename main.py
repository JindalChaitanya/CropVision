# main.py
import os
import sys

from PyQt6 import QtCore, QtWidgets, QtGui
import backend

THUMBS_PER_PAGE = 10
THUMB_SIZE = 128

class WorkerSignals(QtCore.QObject):
    result = QtCore.pyqtSignal(int, dict)
    error = QtCore.pyqtSignal(int, str)

class DetectWorker(QtCore.QRunnable):
    def __init__(self, idx, image_path, threshold, target_class):
        super().__init__()
        self.idx = idx
        self.image_path = image_path
        self.threshold = threshold
        self.target_class = target_class
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            detections = backend.detect_objects(
                self.image_path, self.threshold, self.target_class
            )
            self.signals.result.emit(self.idx, detections)
        except Exception as e:
            self.signals.error.emit(self.idx, str(e))

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DETR Browser")
        self.threadpool = QtCore.QThreadPool.globalInstance()

        # Ensure a model is loaded initially
        backend.init_model()

        # State
        self.src_dir = ""
        self.dest_dir = ""
        self.image_paths = []
        self.page = 0
        self.current_idx = None
        self.page_threshold = 0.5

        # UI Setup
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Directory selectors
        form = QtWidgets.QFormLayout()
        self.srcEdit = QtWidgets.QLineEdit()
        self.browseSrcBtn = QtWidgets.QPushButton("Browse Source...")
        self.destEdit = QtWidgets.QLineEdit()
        self.browseDestBtn = QtWidgets.QPushButton("Browse Destination...")
        form.addRow("Source Dir:", self.srcEdit)
        form.addRow("", self.browseSrcBtn)
        form.addRow("Destination Dir:", self.destEdit)
        form.addRow("", self.browseDestBtn)
        layout.addLayout(form)

        # Model selector
        model_layout = QtWidgets.QHBoxLayout()
        model_name = backend._model.config.name_or_path if hasattr(backend, '_model') else ''
        self.modelEdit = QtWidgets.QLineEdit(model_name)
        self.modelUpdateBtn = QtWidgets.QPushButton("Load Model")
        model_layout.addWidget(QtWidgets.QLabel("Model:"))
        model_layout.addWidget(self.modelEdit)
        model_layout.addWidget(self.modelUpdateBtn)
        layout.addLayout(model_layout)

        # Pagination and page-level controls
        page_ctrl = QtWidgets.QHBoxLayout()
        self.prevBtn = QtWidgets.QPushButton("Previous")
        self.nextBtn = QtWidgets.QPushButton("Next")
        self.pageConfSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.pageConfSlider.setRange(1, 100)
        self.pageConfSlider.setValue(int(self.page_threshold * 100))
        self.pageConfLabel = QtWidgets.QLabel(f"{self.page_threshold:.2f}")
        page_ctrl.addWidget(self.prevBtn)
        page_ctrl.addWidget(self.nextBtn)
        page_ctrl.addWidget(QtWidgets.QLabel("Page Threshold:"))
        page_ctrl.addWidget(self.pageConfSlider)
        page_ctrl.addWidget(self.pageConfLabel)
        layout.addLayout(page_ctrl)

        # Thumbnails
        self.thumbList = QtWidgets.QListWidget()
        self.thumbList.setIconSize(QtCore.QSize(THUMB_SIZE, THUMB_SIZE))
        layout.addWidget(self.thumbList)

        # Action buttons
        action_layout = QtWidgets.QHBoxLayout()
        self.saveSingleBtn = QtWidgets.QPushButton("Save Current Crop")
        self.savePageBtn = QtWidgets.QPushButton("Save All Crops (Page)")
        action_layout.addWidget(self.saveSingleBtn)
        action_layout.addWidget(self.savePageBtn)
        layout.addLayout(action_layout)

        # Preview + controls
        bottom = QtWidgets.QHBoxLayout()
        left = QtWidgets.QVBoxLayout()
        self.previewLabel = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.previewLabel)
        self.deleteBtn = QtWidgets.QPushButton("Delete Image")
        self.detectBtn = QtWidgets.QPushButton("Detect Object")
        left.addWidget(self.detectBtn)
        left.addWidget(self.deleteBtn)
        bottom.addLayout(left)

        right = QtWidgets.QFormLayout()
        self.classCombo = QtWidgets.QComboBox()
        self.classCombo.addItem("All")
        self.classCombo.addItems(backend.get_labels())
        right.addRow("Class Filter:", self.classCombo)
        bottom.addLayout(right)

        layout.addLayout(bottom)

    def _connect_signals(self):
        self.browseSrcBtn.clicked.connect(self._browse_src)
        self.browseDestBtn.clicked.connect(self._browse_dest)
        self.modelUpdateBtn.clicked.connect(self._load_model)
        self.srcEdit.editingFinished.connect(self._load_images)
        self.prevBtn.clicked.connect(self._prev_page)
        self.nextBtn.clicked.connect(self._next_page)
        self.pageConfSlider.valueChanged.connect(self._on_page_conf_change)
        self.thumbList.itemClicked.connect(self._on_thumb_clicked)
        self.saveSingleBtn.clicked.connect(self._save_single)
        self.savePageBtn.clicked.connect(self._save_page)
        self.deleteBtn.clicked.connect(self._on_delete_clicked)
        self.detectBtn.clicked.connect(self._on_detect_clicked)

    def _browse_src(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Source Directory")
        if d:
            self.srcEdit.setText(d)
            self._load_images()

    def _browse_dest(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Destination Directory")
        if d:
            self.destEdit.setText(d)

    def _load_model(self):
        model_name = self.modelEdit.text().strip()
        if model_name:
            backend.init_model(model_name)

    def _load_images(self):
        self.src_dir = self.srcEdit.text()
        if os.path.isdir(self.src_dir):
            self.image_paths = backend.list_images(self.src_dir)
            self.page = 0
            self._refresh_thumbs()

    def _refresh_thumbs(self):
        self.thumbList.clear()
        start = self.page * THUMBS_PER_PAGE
        for idx, path in enumerate(self.image_paths[start:start+THUMBS_PER_PAGE]):
            thumb = QtGui.QPixmap(path).scaled(
                THUMB_SIZE, THUMB_SIZE, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
            item = QtWidgets.QListWidgetItem(QtGui.QIcon(thumb), os.path.basename(path))
            self.thumbList.addItem(item)

    def _prev_page(self):
        if self.page > 0:
            self.page -= 1
            self._refresh_thumbs()

    def _next_page(self):
        if (self.page + 1) * THUMBS_PER_PAGE < len(self.image_paths):
            self.page += 1
            self._refresh_thumbs()

    def _on_page_conf_change(self, val):
        self.page_threshold = val / 100.0
        self.pageConfLabel.setText(f"{self.page_threshold:.2f}")

    def _on_thumb_clicked(self, item):
        idx = self.thumbList.row(item) + self.page * THUMBS_PER_PAGE
        self.current_idx = idx
        path = self.image_paths[idx]
        pix = QtGui.QPixmap(path).scaled(
            400, 400, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self.previewLabel.setPixmap(pix)

    def _on_detect_clicked(self):
        if self.current_idx is None:
            return
        idx = self.current_idx
        path = self.image_paths[idx]
        thr = self.page_threshold
        cls = None if self.classCombo.currentText() == "All" else self.classCombo.currentText()
        worker = DetectWorker(idx, path, thr, cls)
        worker.signals.result.connect(self._on_detect_result)
        worker.signals.error.connect(self._on_detect_error)
        self.threadpool.start(worker)

    def _on_detect_result(self, idx, detections):
        path = self.image_paths[idx]
        pix = QtGui.QPixmap(path).scaled(
            400, 400, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        painter = QtGui.QPainter(pix)
        pen = QtGui.QPen(QtGui.QColor("red"))
        pen.setWidth(2)
        painter.setPen(pen)
        orig = QtGui.QImage(path)
        sx = pix.width() / orig.width()
        sy = pix.height() / orig.height()
        for box in detections["boxes"]:
            x1, y1, x2, y2 = box.tolist()
            painter.drawRect(x1*sx, y1*sy, (x2-x1)*sx, (y2-y1)*sy)
        painter.end()
        self.previewLabel.setPixmap(pix)

    def _on_detect_error(self, idx, errmsg):
        QtWidgets.QMessageBox.critical(self, "Detection Error", errmsg)

    def _save_single(self):
        if self.current_idx is None or not self.destEdit.text():
            return
        path = self.image_paths[self.current_idx]
        detections = backend.detect_objects(
            path, self.page_threshold,
            None if self.classCombo.currentText() == "All" else self.classCombo.currentText()
        )
        backend.crop_and_save(
            path, detections, self.destEdit.text(),
            prefix=os.path.splitext(os.path.basename(path))[0]
        )

    def _save_page(self):
        if not self.destEdit.text():
            return
        start = self.page * THUMBS_PER_PAGE
        for path in self.image_paths[start:start+THUMBS_PER_PAGE]:
            detections = backend.detect_objects(
                path, self.page_threshold,
                None if self.classCombo.currentText() == "All" else self.classCombo.currentText()
            )
            backend.crop_and_save(
                path, detections, self.destEdit.text(),
                prefix=os.path.splitext(os.path.basename(path))[0]
            )

    def _on_delete_clicked(self):
        if self.current_idx is None:
            return
        path = self.image_paths[self.current_idx]
        resp = QtWidgets.QMessageBox.question(
            self, "Confirm Delete", f"Delete {os.path.basename(path)}?"
        )
        if resp == QtWidgets.QMessageBox.StandardButton.Yes:
            os.remove(path)
            del self.image_paths[self.current_idx]
            self.current_idx = None
            self._refresh_thumbs()
            self.previewLabel.clear()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.resize(1000, 700)
    window.show()
    sys.exit(app.exec())