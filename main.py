import PySide6.QtWidgets as qtw
import sys

import mainWindow

def main():
    app = qtw.QApplication(sys.argv)
    window = mainWindow.MainWindow()
    window.show()
    sys.exit(app.exec())
    return

if __name__ == "__main__":
    main()