import PySide6.QtWidgets as qtw
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import sys

class TimeInputArea(qtw.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.createComponents()

    def createComponents(self):
        layout = qtw.QGridLayout(self)

        self.timeInputLineEdit = qtw.QLineEdit(parent=self)
        addButton = qtw.QPushButton("追加", self)
        self.timeSelectComboBox = qtw.QComboBox(parent=self)

        layout.addWidget(self.timeInputLineEdit, 0, 0)
        layout.addWidget(addButton, 0, 1)
        layout.addWidget(self.timeSelectComboBox, 1, 0, 1, 2)

    def addTime(self, times:float):
        self.timeSelectComboBox.addItem(str(times))


class GraphArea(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent, background="w")

