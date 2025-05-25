import sys
import os
import logging
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

# Ensure the project root is in the Python path for imports to work
# This might be needed if running from a different directory.
# If running 'python crop_vision/main.py' from 'crop-vision/', it should work.
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crop_vision.gui.main_window import MainWindow
from crop_vision import config

def setup_logging():
    """Configures the logging for the application."""
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format=config.LOG_FORMAT, stream=sys.stdout)
    # You could also add a FileHandler here to log to a file
    logging.getLogger("ultralytics").setLevel(logging.WARNING) # Reduce YOLO spam
    logging.getLogger("PIL").setLevel(logging.WARNING) # Reduce Pillow spam
    log = logging.getLogger(__name__)
    log.info("Logging configured.")
    return log

def main():
    """Main function to setup and run the application."""
    log = setup_logging()
    log.info("Starting Crop Vision Application...")

    app = QApplication(sys.argv)

    # Set Application Icon (Optional)
    icon_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'icon.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        log.info(f"Loaded icon from {icon_path}")
    else:
        log.warning(f"Icon file not found at {icon_path}")

    # Apply a simple stylesheet (optional)
    app.setStyleSheet("""
        QWidget { font-size: 10pt; }
        QPushButton { padding: 6px; }
        QLineEdit { padding: 3px; }
        QListWidget { border: 1px solid #ccc; background-color: #f0f0f0; }
        QLabel#image_preview_label { background-color: #2c2c2c; } /* Specific ID */
        QMessageBox { min-width: 300px; }
    """)
    log.debug("Stylesheet applied.")

    main_win = MainWindow()
    main_win.show()
    log.info("Main window shown.")

    try:
        sys.exit(app.exec())
    except Exception as e:
        log.critical(f"Application exited with an error: {e}", exc_info=True)

if __name__ == '__main__':
    main()