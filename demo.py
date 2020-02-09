import sys
from PyQt5.QtWidgets import QApplication
import UI_manager

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = UI_manager.Login()
    window.show()
    sys.exit(app.exec())