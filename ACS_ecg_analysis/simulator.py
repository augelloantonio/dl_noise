"""
Simulation to test the filtering function and to test 
the arrhythmias script functioning on a real time database connection data coming
"""

import os
import sys
import time
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
import matplotlib
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from datetime import datetime, timedelta
import threading
import pandas as pd
from ACS_ecg_analysis.ecg_processing.signal_filtering import movingAverageMean, artifactRemoval, bandpassFilt, derivateStep, movingAverageMeanPamTompkins

sampling_rate = 250

###################################################################
#                                                                 #
#                    PLOT A LIVE GRAPH (PyQt5)                    #
#                  -----------------------------                  #
#                                                                 #
###################################################################

matplotlib.use("Qt5Agg")

n = 0
x_lines = []
y_lines = []

while n <= 1200:
    # n += 10
    n+=80
    x_lines.append(n)

l = -1
while l <= 4:
    l+= 0.5
    y_lines.append(l)

class CustomMainWindow(QMainWindow):
    def __init__(self):
        super(CustomMainWindow, self).__init__()
        # Define the geometry of the main window
        # self.setGeometry(600, 600, 1600, 800)
        self.setWindowTitle("Real Time ECG")
        # Create FRAME_A
        self.FRAME_A = QFrame(self)
        self.FRAME_A.setStyleSheet(
            "QWidget { background-color: %s }" % QColor(210, 210, 235, 255).name())
        self.LAYOUT_A = QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        self.setCentralWidget(self.FRAME_A)
        # Place the zoom button
        """ self.zoomBtn = QPushButton(text='zoom')
        self.zoomBtn.setFixedSize(100, 50)
        self.zoomBtn.clicked.connect(self.zoomBtnAction)
        self.LAYOUT_A.addWidget(self.zoomBtn, *(0, 0)) """
        # Place the matplotlib figure
        self.myFig = CustomFigCanvas()
        self.LAYOUT_A.addWidget(self.myFig, *(0, 1))
        # Add the callbackfunc to ..
        myDataLoop = threading.Thread(
            name='myDataLoop', target=dataSendLoop, daemon=True, args=(self.addData_callbackFunc,))
        myDataLoop.start()
        self.show()
        return

    """ def zoomBtnAction(self):
        print("zoom in")
        self.myFig.zoomIn(0.01)
        return """

    def addData_callbackFunc(self, value):
        # print("Add data: " + str(value))
        self.myFig.addData(value)
        return

''' End Class '''


class CustomFigCanvas(FigureCanvas, TimedAnimation):

    def __init__(self):

        # To set pending on +- 0.2 on trace range
        # To get average use np.average(filtered)
        # min_reader = 0.353
        # max_reader = 0.392

        min_reader = 0.16
        max_reader = 0.23
        self.addedData = []
        print(matplotlib.__version__)

        # The data
        self.xlim = 1200
        self.n = np.linspace(0, self.xlim - 1, self.xlim)
        a = []
        b = []
        a.append(2.0)
        a.append(4.0)
        a.append(2.0)
        b.append(4.0)
        b.append(3.0)
        b.append(4.0)
        self.y = (self.n * 0.0) + 50
        # The window
        self.fig = Figure(figsize=(40, 40), dpi=100)
        self.ax1 = self.fig.add_subplot(111)
        # self.ax1 settings
        self.ax1.set_xlabel('80 = 200msec (5mm)')
        self.ax1.set_ylabel('5mm = 0.5mV')
        self.line1 = Line2D([], [], color='blue')
        self.line1_tail = Line2D([], [], color='red', linewidth=2)
        self.line1_head = Line2D(
            [], [], color='red', marker='o', markeredgecolor='r')
        self.ax1.add_line(self.line1)
        self.ax1.add_line(self.line1_tail)
        self.ax1.add_line(self.line1_head)
        self.ax1.set_xlim(0, self.xlim - 1)
        # self.ax1.set_ylim(-0.005, 0.005)
        # self.ax1.set_ylim(-0.01, 0.01) # Isabella
        # self.ax1.set_ylim(, 0.065) # Virginia & Luciano sample
        # self.ax1.set_ylim(-0.05, 0.04) # Pasqualina (too largne anyway)
        # self.ax1.set_ylim(-0.005, 0.015) # Eleonora
        # self.ax1.set_ylim(0.04, 0.07) # Giovanni sample
        # self.ax1.set_ylim(400, 700)
        # self.ax1.set_ylim(-250, 250.11)
        self.ax1.set_ylim(-1, 2.0)
        # self.ax1.set_ylim(-800, 850)
        # self.ax1.set_ylim(0.045, 0.06) # first sample
        # self.ax1.set_ylim(-0.006, 0.01)
        #self.ax1.set_xticks(x_lines)
        #self.ax1.xaxis.grid(True, color='r')
        #self.ax1.yaxis.grid(True, color='r')
        #self.ax1.set_yticks(y_lines)
        #self.ax1.axes.xaxis.set_ticklabels([])
        #self.ax1.yaxis.set_ticklabels([])

        self.ax1.legend(['Signal','1mm = 0.04sec'])
        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval=4, blit=True)
        self.ax1.axes.relim()
        self.ax1.axes.autoscale_view(True,True,True)
        return

    def new_frame_seq(self):
        return iter(range(self.n.size))

    def _init_draw(self):
        lines = [self.line1, self.line1_tail, self.line1_head]
        for l in lines:
            l.set_data([], [])
        return

    def addData(self, value):
        self.addedData.append(value)
        return

    def updateData(self, value):
        self.addedData.append(value)

    def _step(self, *args):
        # Extends the _step() method for the TimedAnimation class.
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            self.abc += 1
            print(str(self.abc))
            TimedAnimation._stop(self)
            pass
        return

    def _draw_frame(self, framedata):
        margin = 2
        while(len(self.addedData) > 0):
            self.y = np.roll(self.y, -1)
            self.y[-1] = self.addedData[0]
            del(self.addedData[0])

        self.line1.set_data(
            self.n[0: self.n.size - margin], self.y[0: self.n.size - margin])
        self.line1_tail.set_data(np.append(
            self.n[-10:-1 - margin], self.n[-1 - margin]), np.append(self.y[-10:-1 - margin], self.y[-1 - margin]))
        self.line1_head.set_data(self.n[-1 - margin], self.y[-1 - margin])
        self._drawn_artists = [self.line1, self.line1_tail, self.line1_head]
        self.ax1.set_ylim(-100+(np.min(self.y[0: self.n.size - margin])), 100+(np.max(self.y[0: self.n.size - margin])))
        return


''' End Class '''


# You need to setup a signal slot mechanism, to
# send data to your GUI in a thread-safe way.
class Communicate(QObject):
    data_signal = pyqtSignal(float)


''' End Class '''


def dataSendLoop(addData_callbackFunc):
    # Setup the signal-slot mechanism.
    mySrc = Communicate()
    mySrc.data_signal.connect(addData_callbackFunc)

    ecg = np.genfromtxt("/Users/antonioaugello/Downloads/drive-download-20210914T100904Z-001/Raysi Keyvan/20210914_102001.ecg", delimiter=',')[:15000]
    ecg = ecg[~np.isnan(ecg)]
    ecg= np.multiply(ecg, 100)

    ecg = movingAverageMean(ecg, 125)

##############################################################################
    if not isinstance(ecg, np.ndarray):
        ecg = np.array(ecg)

    # get the ecg data
    n = np.linspace(0, 499, 500)


    i = 0


    # Loop over the signal 
    while True:
        y = ecg
        time.sleep(0.004)
        try:
            mySrc.data_signal.emit(y[i])  # <- Here you emit a signal!
        except:
            print(i)
            print(y[i])
            print('Error in signal received')
            break
        i += 1
    ###
###

if __name__ == '__main__':
    QApplication.setStyle(QStyleFactory.create('fusion'))
    app = QApplication(sys.argv)
    display = CustomMainWindow()
    sys.exit(app.exec_())