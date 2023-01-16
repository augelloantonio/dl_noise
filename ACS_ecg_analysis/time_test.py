import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from ACS_ecg_analysis.ecg_processing.signal_anomalies_detector import beatsClassification, countAnomalies, beatsClassification2, detect_arrythmia, detect_afib
from ACS_ecg_analysis.ecg_processing.signal_filtering import movingAverageMean, artifactRemoval, bandpassFilt, derivateStep, movingAverageMeanPamTompkins, newfilter
from ACS_ecg_analysis.ecg_processing.signal_peak_detector import ACSPeakDetector, panPeakDetect, newACSPeakDetector2, ACSPeakDetector3, newACSPeakDetector
from ACS_ecg_analysis.ecg_processing.signal_analysis import calculateNNI
from ACS_ecg_analysis.mit_processing.mit_reader import getAnnotation, countAnnotationAnomalies
from ACS_ecg_analysis.mit_processing.load_annotation import loadAnnotationSample 
from ACS_ecg_analysis.mit_processing.mit_analysis import checkNegative, checkPositive
import os
import time
from pam import panTompkins

ecg = np.genfromtxt("/Users/antonioaugello/Desktop/projects/ecg_analisys/data/mit/converted/file_out_102.dat.csv", delimiter=',')
ecg = ecg[~np.isnan(ecg)]
    
newEcgFilt = bandpassFilt(ecg, 4, 360, 15, 5)

derivateSignal = derivateStep(newEcgFilt)

squaredEcgfromderivate = np.power(np.abs(derivateSignal), 2)

panTompkinsEcgfromderivate = movingAverageMeanPamTompkins(squaredEcgfromderivate, 360)  

start_time = time.time()
peaks, thrheshold_list = ACSPeakDetector3(panTompkinsEcgfromderivate, 360)

print("--- %s seconds ---" % (time.time() - start_time))
print(len(peaks))

##############
# ANNOTATIONS
##############

""" annotationSample = loadAnnotationSample("file_out_207.dat.csv")
print(len(annotationSample))
print("+")
fakePositive = checkPositive(annotationSample, peaks)
print("-")
fakeNegative = checkNegative(annotationSample, peaks)  """


# Plot Threshold
""" newList=[]
y = 0
z = 0
for i in peaks:
    for x in range(z, i):
        newList.append(thrheshold_list[y])
        z = i
    y += 1 """

##########
## POTTING
##########

ECG = np.array(panTompkinsEcgfromderivate)
plt.plot(panTompkinsEcgfromderivate)
#plt.plot(newList)
#plt.scatter(annotationSample, ECG[annotationSample], c = 'k', s = 30,label='MIT Annotations' )
plt.scatter(peaks, ECG[peaks], marker="x", c = 'g', s = 30, label='Our Detected Peaks')
#plt.vlines(fakePositive, ymin=np.min(ECG), ymax=np.max(ECG), color="y",linewidth=1, label='Fake +')
#plt.vlines(fakeNegative, ymin=np.min(ECG), ymax=np.max(ECG), color="r", linewidth=1, label='Fake -')
plt.legend()
plt.show()