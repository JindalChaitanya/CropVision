from PyQt6.QtCore import QObject, pyqtSignal

class WorkerSignals(QObject):
    """
    Defines signals available from a running worker thread.
    Supported signals:
    - finished: No data
    - error: str (error message)
    - result: object (data returned from worker)
    - progress: int (0-100 or current count)
    - message: str (status messages)
    - batch_item_processed: int (index of item processed)
    """
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    batch_item_processed = pyqtSignal(int, str) # Emits index and status/error message