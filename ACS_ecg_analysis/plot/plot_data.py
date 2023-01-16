import matplotlib.pyplot as plt
import numpy as np

def plotData(raw_data, filtered_data=None, peaks=None):
    ECG = np.array(raw_data)
    plt.plot(raw_data, label='Raw ECG')
    if filtered_data!=None:
        plt.plot(filtered_data, label='Filtered ECG')
    if peaks!=None:
        plt.scatter(peaks, ECG[peaks], marker="x", c = 'g', s = 30, label='Our Detected Peaks')
    plt.legend()
    plt.show()