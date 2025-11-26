import Impedance
import pandas as pd
import pyqtgraph as pg

df = pd.read_table("/Users/fukuchikeita/Documents/実験データ/impedance/250918/250918_F3-5_2mLmin-1_pH2.6_01M_Na2SO4_02_PEIS_C01のコピー2.txt")
z_re = df["Re(Z)/Ohm"]
z_im = df["-Im(Z)/Ohm"]
freq = df["freq/Hz"]
time = df["time/s"]
cycleNumber = df["cycle number"]
impedance = Impedance.Impedance(z_re, z_im, freq)
imp_3D = Impedance.ThreeDimentionImpedance(impedance, time, cycleNumber)
imp_3D.getImpedanceSpilitedByCycle()
imp_3D.getImpedanceSpilitedByFrequency()
imp_3D.getInstantaneoutImpedance(1200)
data = imp_3D.getLinePlotValueToTime(dt=10)

app = pg.mkQApp()
w = pg.PlotWidget()
w.addItem(pg.PlotDataItem(data[1][:, 2], data[1][:, 0]))
w.addItem(pg.ScatterPlotItem(
    imp_3D.getImpedanceSpilitedByFrequency()[1][2],
    imp_3D.getImpedanceSpilitedByFrequency()[1][1].getRealImpedance()
    ))
w.show()
app.exec()