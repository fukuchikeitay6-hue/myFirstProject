import PySide6.QtWidgets as qtw
import widgetsGroup as wg

class MainWindow(qtw.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Impedance analyzer")
        self.showMaximized()
        self.createWidgets()
        
    def createWidgets(self):
        centralWidget = qtw.QWidget(parent=self)
        self.setCentralWidget(centralWidget)
        layout = qtw.QGridLayout(centralWidget)

        self.inputArea = wg.TimeInputArea(self)
        layout.addWidget(self.inputArea)