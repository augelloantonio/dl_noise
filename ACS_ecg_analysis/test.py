from brt_processing.signal_peak_detector import *
from brt_processing.signal_filtering import *
import matplotlib.pyplot as plt
import neurokit2 as nk
from ACS_ecg_analysis.ecg_processing.signal_anomalies_detector import beatsClassification, countAnomalies, beatsClassification2, detect_arrythmia, detect_afib
from ACS_ecg_analysis.ecg_processing.signal_filtering import movingAverageMean, artifactRemoval, bandpassFilt, derivateStep, movingAverageMeanPamTompkins, newfilter
from ACS_ecg_analysis.ecg_processing.signal_peak_detector import ACSPeakDetector, panPeakDetect, newACSPeakDetector, newACSPeakDetector2
from ACS_ecg_analysis.ecg_processing.signal_analysis import calculateNNI, calculateBpm


ecg = np.genfromtxt("/Users/antonioaugello/Downloads/20210927_150407.ecg", delimiter=',')
ecg = ecg[~np.isnan(ecg)] 

fs = 250

brt = np.genfromtxt("/Users/antonioaugello/Downloads/20210927_150407.brt", delimiter=',')

lowPassRemoved = Butterworth_Lowpass(brt, 4, 250, 4)
meanRemoved = movingAverageMeanBrt(lowPassRemoved)

newbrtpeaks = findPeakSignal(meanRemoved)

signal_average_mean_removed = movingAverageMean(ecg, fs)
finalEcgArtifactRemoved = artifactRemoval(signal_average_mean_removed)

newEcgFilt = bandpassFilt(finalEcgArtifactRemoved, 4, fs, 15, 5)

derivateSignal = derivateStep(newEcgFilt)

squaredEcgfromderivate = np.power(np.abs(derivateSignal), 2)

panTompkinsEcgfromderivate = movingAverageMeanPamTompkins(
    squaredEcgfromderivate, fs) 

peaks,thrheshold_list = newACSPeakDetector2(panTompkinsEcgfromderivate, 360)

# Plot Threshold
newList=[]
y = 0
z = 0
for i in peaks:
    for x in range(z, i):
        newList.append(thrheshold_list[y])
        z = i
    y += 1

#plt.plot(meanRemoved, label='meanRemoved')
plt.plot(newbrtpeaks, label='peaks python')
ECG = np.array(panTompkinsEcgfromderivate)
#plt.plot(panTompkinsEcgfromderivate)
plt.plot(newList)
#plt.scatter(peaks, ECG[peaks], marker="x", c = 'g', s = 30, label='Our Detected Peaks')
plt.legend()
plt.show()