# main.py
import os
import sys
import time
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtGui import QImage
import backend

THUMBS_PER_PAGE = 10
THUMB_SIZE = 128

class WorkerSignals(QtCore.QObject):
    result = QtCore.pyqtSignal(int, dict)
    error = QtCore.pyqtSignal(int, str)

class DetectWorker(QtCore.QRunnable):
    def __init__(self, idx, path, threshold, cls, model_name):
        super().__init__()
        self.idx = idx
        self.path = path
        self.threshold = threshold
        self.target_class = cls
        self.model_name = model_name
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            det = backend.detect_objects(
                self.path, self.threshold, self.target_class, self.model_name
            )
            self.signals.result.emit(self.idx, det)
        except Exception as e:
            self.signals.error.emit(self.idx, str(e))

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DETR Browser")
        self.resize(1000, 700)
        self.threadpool = QtCore.QThreadPool.globalInstance()
        backend.init_model()
        # State
        self.src_dir = ""
        self.dest_dir = ""
        self.model_name = "facebook/detr-resnet-50"
        self.image_paths = []
        self.page = 0
        self.current_idx = None
        self.page_threshold = 0.5
        # Build UI
        self._build_ui()
        self._connect_signals()
        self._load_labels()

    def _build_ui(self):
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        v = QtWidgets.QVBoxLayout(w)
        form = QtWidgets.QFormLayout()
        self.srcEdit = QtWidgets.QLineEdit()
        self.browseSrc = QtWidgets.QPushButton("Browse Source...")
        self.destEdit = QtWidgets.QLineEdit()
        self.browseDest = QtWidgets.QPushButton("Browse Dest...")
        form.addRow("Source Dir:", self.srcEdit)
        form.addRow("", self.browseSrc)
        form.addRow("Dest Dir:", self.destEdit)
        form.addRow("", self.browseDest)
        v.addLayout(form)
        mh = QtWidgets.QHBoxLayout()
        self.modelEdit = QtWidgets.QLineEdit(self.model_name)
        self.modelLoad = QtWidgets.QPushButton("Load Model")
        mh.addWidget(QtWidgets.QLabel("Model:"))
        mh.addWidget(self.modelEdit)
        mh.addWidget(self.modelLoad)
        v.addLayout(mh)
        ph = QtWidgets.QHBoxLayout()
        self.prevBtn = QtWidgets.QPushButton("Previous")
        self.nextBtn = QtWidgets.QPushButton("Next")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(1, 100)
        self.slider.setValue(50)
        self.lblThreshold = QtWidgets.QLabel("0.50")
        ph.addWidget(self.prevBtn)
        ph.addWidget(self.nextBtn)
        ph.addWidget(QtWidgets.QLabel("Threshold:"))
        ph.addWidget(self.slider)
        ph.addWidget(self.lblThreshold)
        v.addLayout(ph)
        self.thumbList = QtWidgets.QListWidget()
        self.thumbList.setIconSize(QtCore.QSize(THUMB_SIZE, THUMB_SIZE))
        self.thumbList.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        v.addWidget(self.thumbList)
        ah = QtWidgets.QHBoxLayout()
        self.saveOne = QtWidgets.QPushButton("Save Current Crop")
        self.savePage = QtWidgets.QPushButton("Save Page Crops")
        ah.addWidget(self.saveOne)
        ah.addWidget(self.savePage)
        v.addLayout(ah)
        bh = QtWidgets.QHBoxLayout()
        lh = QtWidgets.QVBoxLayout()
        self.preview = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        lh.addWidget(self.preview)
        self.detectBtn = QtWidgets.QPushButton("Detect")
        self.deleteBtn = QtWidgets.QPushButton("Delete")
        lh.addWidget(self.detectBtn)
        lh.addWidget(self.deleteBtn)
        bh.addLayout(lh)
        rh = QtWidgets.QFormLayout()
        self.classCombo = QtWidgets.QComboBox()
        rh.addRow("Class:", self.classCombo)
        bh.addLayout(rh)
        v.addLayout(bh)

    def _connect_signals(self):
        self.browseSrc.clicked.connect(self._browse_src)
        self.browseDest.clicked.connect(self._browse_dest)
        self.modelLoad.clicked.connect(self._update_model)
        self.srcEdit.editingFinished.connect(self._load_images)
        self.prevBtn.clicked.connect(self._prev_page)
        self.nextBtn.clicked.connect(self._next_page)
        self.slider.valueChanged.connect(self._on_slider)
        self.thumbList.itemClicked.connect(self._on_thumb)
        self.detectBtn.clicked.connect(self._detect)
        self.saveOne.clicked.connect(self._save_current)
        self.savePage.clicked.connect(self._save_page)
        self.deleteBtn.clicked.connect(self._delete)

    def _load_labels(self):
        self.classCombo.clear()
        labels = backend.get_labels(self.model_name)
        self.classCombo.addItem("All")
        self.classCombo.addItems(sorted(labels))

    def _browse_src(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Source")
        if d:
            self.srcEdit.setText(d)
            self._load_images()

    def _browse_dest(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Dest")
        if d:
            self.destEdit.setText(d)

    def _update_model(self):
        mp = self.modelEdit.text().strip()
        if mp:
            self.model_name = mp
            backend.init_model(mp)
            self._load_labels()

    def _load_images(self):
        src = self.srcEdit.text()
        if os.path.isdir(src):
            self.image_paths = backend.list_images(src)
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
            item.setData(QtCore.Qt.ItemDataRole.UserRole, start+idx)
            self.thumbList.addItem(item)

    def _prev_page(self):
        if self.page > 0:
            self.page -= 1
            self._refresh_thumbs()

    def _next_page(self):
        if (self.page+1)*THUMBS_PER_PAGE < len(self.image_paths):
            self.page += 1
            self._refresh_thumbs()

    def _on_slider(self, val):
        self.page_threshold = val / 100.0
        self.lblThreshold.setText(f"{self.page_threshold:.2f}")

    def _on_thumb(self, item):
        idx = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self.current_idx = idx
        path = self.image_paths[idx]
        pix = QtGui.QPixmap(path).scaled(
            400, 400, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self.preview.setPixmap(pix)

    def _detect(self):
        if self.current_idx is None:
            return
        worker = DetectWorker(
            self.current_idx,
            self.image_paths[self.current_idx],
            self.page_threshold,
            None if self.classCombo.currentText()=="All" else self.classCombo.currentText(),
            self.model_name
        )
        worker.signals.result.connect(self._on_detect_result)
        worker.signals.error.connect(self._on_detect_error)
        self.threadpool.start(worker)

    def _on_detect_result(self, idx, det):
        path = self.image_paths[idx]
        pix = QtGui.QPixmap(path).scaled(
            400, 400, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        painter = QtGui.QPainter(pix)
        pen = QtGui.QPen(QtGui.QColor('red'))
        pen.setWidth(2)
        painter.setPen(pen)
        orig = QImage(path)
        sx = pix.width() / orig.width()
        sy = pix.height() / orig.height()
        for box in det['boxes']:
            x1, y1, x2, y2 = box.tolist()
            ix = int(x1 * sx)
            iy = int(y1 * sy)
            iw = int((x2 - x1) * sx)
            ih = int((y2 - y1) * sy)
            painter.drawRect(ix, iy, iw, ih)
        painter.end()
        self.preview.setPixmap(pix)

    def _on_detect_error(self, idx, msg):
        QtWidgets.QMessageBox.critical(self, "Error", msg)

    def _save_current(self):
        if self.current_idx is None or not self.destEdit.text():
            return
        path = self.image_paths[self.current_idx]
        det = backend.detect_objects(
            path, self.page_threshold,
            None if self.classCombo.currentText()=="All" else self.classCombo.currentText(),
            self.model_name
        )
        prefix = os.path.splitext(os.path.basename(path))[0]
        backend.crop_and_save(path, det, self.destEdit.text(), prefix)

    def _save_page(self):
        if not self.destEdit.text():
            return
        start = self.page*THUMBS_PER_PAGE
        for p in self.image_paths[start:start+THUMBS_PER_PAGE]:
            det = backend.detect_objects(
                p, self.page_threshold,
                None if self.classCombo.currentText()=="All" else self.classCombo.currentText(),
                self.model_name
            )
            prefix = os.path.splitext(os.path.basename(p))[0]
            backend.crop_and_save(p, det, self.destEdit.text(), prefix)

    def _delete(self):
        if self.current_idx is None:
            return
        path = self.image_paths[self.current_idx]
        if QtWidgets.QMessageBox.question(self, "Delete?", f"Delete {os.path.basename(path)}?") == QtWidgets.QMessageBox.StandardButton.Yes:
            os.remove(path)
            del self.image_paths[self.current_idx]
            self.current_idx = None
            self._refresh_thumbs()
            self.preview.clear()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())