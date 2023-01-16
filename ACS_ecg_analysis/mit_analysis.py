import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from ecg_processing.signal_anomalies_detector import beatsClassification, countAnomalies, beatsClassification2, detect_arrythmia, detect_afib
from ecg_processing.signal_filtering import movingAverageMean, artifactRemoval, bandpassFilt, derivateStep, movingAverageMeanPamTompkins, newfilter
from ecg_processing.signal_peak_detector import ACSPeakDetector, panPeakDetect, newACSPeakDetector, newACSPeakDetector2, ACSPeakDetector3
from ecg_processing.signal_analysis import calculateNNI, calculateRmssd
from mit_processing.mit_reader import getAnnotation, countAnnotationAnomalies
from mit_processing.load_annotation import loadAnnotationSample 
from mit_processing.mit_analysis import checkNegative, checkPositive
import os
import time
from pam import panTompkins

fileList = []
for filename in os.listdir("/Users/antonioaugello/Desktop/projects/ecg_analisys/data/mit/converted/"):
    fileList.append(filename)

fileList.sort()

for f in fileList:

    #print("---------------")
    print(f)
    ecg = np.genfromtxt("/Users/antonioaugello/Desktop/projects/ecg_analisys/data/mit/converted/" + f, delimiter=',')
    ecg = ecg[~np.isnan(ecg)]

    newEcgFilt = bandpassFilt(ecg, 4, 360, 15, 5)

    derivateSignal = derivateStep(newEcgFilt)

    squaredEcgfromderivate = np.power(np.abs(derivateSignal), 2)

    panTompkinsEcgfromderivate = movingAverageMeanPamTompkins(squaredEcgfromderivate, 360)
    #start_time = time.time()
    peaks, _ = ACSPeakDetector3(panTompkinsEcgfromderivate, 360)
    #print(time.time() - start_time)
    
    #print(len(peaks))

    nni = calculateNNI(peaks)
    calculateRmssd(nni)
    """beats_annotation = []

    pulse = 0
    beatClassificationList = beatsClassification(nni, peaks, pulse) """
    
    #detect_afib(panTompkinsEcgfromderivate, 250)

    '''x = np.arange(0,len(nni),1)
    y = np.array(nni)

    plt.scatter(x, y)
    plt.show() '''
    
    # "peaks detected = "
    #print(len(peaks))

    ##############
    # ANNOTATIONS
    ##############

    annotationSample = loadAnnotationSample(f)
    # fp = checkPositive(annotationSample, peaks)
    # fn = checkNegative(annotationSample, peaks)  
    
    """ plt.plot = nk.events_plot([peaks], ecg)
    plt.show() """
    
    """ i=0
    while i < len(panTompkinsEcgfromderivate):
        if i%108000==0:
            peaks, _ = ACSPeakDetector3(panTompkinsEcgfromderivate[i-108000:i], 360)
            nni = calculateNNI(peaks)
            # bpm, r = calculateBpm(nni)
            calculateRmssd(nni)
            # total_r.append(bpm)
            # plotData(finalEcgArtifactRemoved[i-108000:i], peaks=peaks)
        i+=1  """

    #countAnomalies(beatClassificationList, peaks)
    #print("______________________________")
    #print()