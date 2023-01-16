import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import wfdb
import math
import datetime
from ACS_ecg_analysis.ecg_processing.signal_anomalies_detector import beatsClassification, countAnomalies, beatsClassification2, detect_arrythmia, detect_afib
from ACS_ecg_analysis.ecg_processing.signal_filtering import movingAverageMean, artifactRemoval, bandpassFilt, movingAverageMeanPamTompkins
from ACS_ecg_analysis.ecg_processing.signal_peak_detector import ACSPeakDetector3
from ACS_ecg_analysis.ecg_processing.signal_analysis import calculateNNI

################
#
# AUTOMATE ANALYSIS
#
################

def calculatePercentage(file):
    totalpercentageRemoved = 0
    durationArtifacts = 0
    totfiles = 0
    durationEcg = 0
    percentageRemoved = 0
    totdurationsample = 0
    signal_quality = {}
    
    ecg = np.genfromtxt(file, delimiter=' ')
    ecg = ecg[~np.isnan(ecg)]

    if len(ecg) > 25000:
        signal_average_mean_removed = movingAverageMean(ecg, 5)
        finalEcgArtifactRemoved = artifactRemoval(signal_average_mean_removed, fs)
    
        count = 0
        for i in finalEcgArtifactRemoved:
            if i == 0:
                count += 1

        percentageRemoved = ((len(ecg)-count)/len(ecg))*100
        if percentageRemoved > 35:
            durationEcg = str(datetime.timedelta(seconds=len(ecg)/250))
            durationArtifacts = str(datetime.timedelta(seconds=count/250))
            totfiles += 1
            
            signal_quality = {"duration":durationEcg, "good_percentage":percentageRemoved, "artifact_duration":durationArtifacts }
    
    return signal_quality

def calculatePercentageFromData(ecg, fs):
    totalpercentageRemoved = 0
    durationArtifacts = 0
    totfiles = 0
    durationEcg = 0
    percentageRemoved = 0
    totdurationsample = 0
    signal_quality = {}
    
    if len(ecg) > 25000:
        signal_average_mean_removed = movingAverageMean(ecg, 5)
        finalEcgArtifactRemoved = artifactRemoval(signal_average_mean_removed, fs)

        count = 0
        for i in finalEcgArtifactRemoved:
            if i == 0:
                count += 1

        percentageRemoved = ((len(ecg)-count)/len(ecg))*100
        durationEcg = str(datetime.timedelta(seconds=len(ecg)/250))
        durationArtifacts = str(datetime.timedelta(seconds=count/250))
        totfiles += 1
            
        signal_quality = {"duration":durationEcg, "good_percentage":percentageRemoved, "artifact_duration":durationArtifacts }
    
    return signal_quality