import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from ACS_ecg_analysis.ecg_processing.signal_anomalies_detector import beatsClassification, countAnomalies, beatsClassification2, detect_arrythmia, detect_afib
from ACS_ecg_analysis.ecg_processing.signal_filtering import movingAverageMean, artifactRemoval, bandpassFilt, derivateStep, movingAverageMeanPamTompkins
from ACS_ecg_analysis.ecg_processing.signal_peak_detector import ACSPeakDetector, newACSPeakDetector
from ACS_ecg_analysis.ecg_processing.signal_analysis import calculateNNI
from ACS_ecg_analysis.mit_processing.mit_reader import getAnnotation, countAnnotationAnomalies
from ACS_ecg_analysis.mit_processing.load_annotation import loadAnnotationSample 
from ACS_ecg_analysis.mit_processing.mit_analysis import checkNegative, checkPositive
import os

fileList = []
for filename in os.listdir("/Users/antonioaugello/Desktop/projects/ecg_analisys/data/sinus_mit/"):
    fileList.append(filename)

fileList.sort()
    
for f in fileList:
    #print("---------------")
    #print(f)
    ecg = np.genfromtxt("/Users/antonioaugello/Desktop/projects/ecg_analisys/data/sinus_mit/" + f, delimiter=',')
    ecg = ecg[~np.isnan(ecg)]

    ecg = ecg[~np.isnan(ecg)]
    
    newEcgFilt = bandpassFilt(ecg, 4, 128, 15, 5)

    derivateSignal = derivateStep(newEcgFilt)

    squaredEcgfromderivate = np.power(np.abs(derivateSignal), 2)

    panTompkinsEcgfromderivate = movingAverageMeanPamTompkins(
        squaredEcgfromderivate, 128)

    peaks = newACSPeakDetector(panTompkinsEcgfromderivate, 128)

    # nni = calculateNNI(peaks)
    beats_annotation = []

    pulse = 0
    #beatClassificationList = beatsClassification(nni, peaks, pulse)
    
    #detect_afib(panTompkinsEcgfromderivate, 250)

    """ x = np.arange(0,len(nni),1)
    y = np.array(nni)

    plt.scatter(x, y)
    plt.show() """
    print(len(peaks))

    ##############
    # ANNOTATIONS
    ##############

    # annotationSample = loadAnnotationSample(filename)
    # checkPositive(annotationSample, peaks)
    # checkNegative(annotationSample, peaks)

    #countAnomalies(beatClassificationList, peaks)
    #print("______________________________")
    #print()
