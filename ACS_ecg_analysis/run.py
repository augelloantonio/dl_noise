import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from ACS_ecg_analysis.plot.plot_data import plotData
from ACS_ecg_analysis.ecg_processing.signal_anomalies_detector import new_tsipouras_2, countAnomalies3, beatsClassification, countAnomalies, beatsClassification2, detect_arrythmia, detect_afib
from ACS_ecg_analysis.ecg_processing.signal_filtering import movingAverageMean, artifactRemoval, bandpassFilt, derivateStep, movingAverageMeanPamTompkins
from ACS_ecg_analysis.ecg_processing.signal_peak_detector import ACSPeakDetector, panPeakDetect, newACSPeakDetector, newACSPeakDetector2, ACSPeakDetector3
from ACS_ecg_analysis.ecg_processing.signal_analysis import calculateNNI, calculateBpm, calculateRmssd, calculateRmssd2
from ACS_ecg_analysis.mit_processing.mit_reader import getAnnotation, countAnnotationAnomalies
from ACS_ecg_analysis.mit_processing.load_annotation import loadAnnotationSample
from ACS_ecg_analysis.mit_processing.mit_analysis import checkNegative, checkPositive
from ACS_ecg_analysis.ecg_processing.signal_time_domain_analysis import get_time_domain_features
from ACS_ecg_analysis.ecg_processing.signal_frequencies_analysis import get_hrv_frequency_measures
from ACS_ecg_analysis.ecg_percentage import calculatePercentageFromData
from scipy.ndimage.filters import gaussian_filter1d


def runAnalysisFromFile(file):
    ecg = np.genfromtxt(file, delimiter=',')
    fs = 250
    signal_average_mean_removed = movingAverageMean(ecg, fs)
    finalEcgArtifactRemoved = artifactRemoval(signal_average_mean_removed)
    newEcgFilt = bandpassFilt(finalEcgArtifactRemoved, 4, fs, 15, 5)
    derivateSignal = derivateStep(newEcgFilt, fs)
    squaredEcgfromderivate = np.power(np.abs(derivateSignal), 2)
    panTompkinsEcgfromderivate = movingAverageMeanPamTompkins(
        squaredEcgfromderivate, fs)
    peaks, thrheshold_list = ACSPeakDetector3(panTompkinsEcgfromderivate, fs)
    nni = calculateNNI(peaks)
    hrv = calculateRmssd(nni)

    bpm, bpm_list = calculateBpm(nni)

    finalBpmList, finalHrvList = calculateIntervalParameters(
        panTompkinsEcgfromderivate, fs)

    beats_annotation = []
    pulse = 0
    i = 0
    while i < len(nni)-1:
        if i % 32 == 0:
            arr_nni = nni[i-32:i]
            normal_rythm = True
            beatClassificationList, pulse = beatsClassification2(
                arr_nni, peaks, pulse, normal_rythm)
            for x in beatClassificationList:
                beats_annotation.append(x)
            detect_afib(arr_nni, fs)
        i += 1

    anomalies, anomalies_time, anomalies_indexing = countAnomalies(
        beats_annotation, peaks)

    parameter = {"bpm_mean": bpm, "bpm_list": finalBpmList,
                    "hrv_mean": hrv, "hrv_list": finalHrvList, "anomalies": anomalies}

    time_domain = get_time_domain_features(nni, bpm_list)

    time_domain = {}
    return parameter, anomalies_time, time_domain, anomalies_indexing, peaks


def runFullAnalysisFromSignal(ecg, fs):
    signal_average_mean_removed = movingAverageMean(ecg, fs)
    finalEcgArtifactRemoved = artifactRemoval(signal_average_mean_removed)
    newEcgFilt = bandpassFilt(finalEcgArtifactRemoved, 4, fs, 15, 5)
    derivateSignal = derivateStep(newEcgFilt, fs)
    squaredEcgfromderivate = np.power(np.abs(derivateSignal), 2)
    panTompkinsEcgfromderivate = movingAverageMeanPamTompkins(
        squaredEcgfromderivate, fs)
    peaks, thrheshold_list = ACSPeakDetector3(panTompkinsEcgfromderivate, fs)
    if len(peaks) < 4:
        return "Error: Not enough peaks found"

    nni = calculateNNI(peaks)

    if len(nni) < 4:
        return "Error: Not enough heart beat found"

    hrv = calculateRmssd(nni)

    bpm, bpm_list = calculateBpm(nni)

    finalBpmList, finalHrvList = calculateIntervalParameters(
        panTompkinsEcgfromderivate, fs)

    beats_annotation = []
    pulse = 0
    normal_rythm = True
    beatClassificationList, pulse = beatsClassification2(
        nni, peaks, pulse, normal_rythm)
    for x in beatClassificationList:
        beats_annotation.append(x)
    # detect_afib(nni, fs)

    anomalies, anomalies_time, anomalies_indexing = countAnomalies(
        beats_annotation, peaks)

    parameter = {"bpm_mean": bpm, "bpm_list": finalBpmList,
                    "hrv_mean": hrv, "hrv_list": finalHrvList, "anomalies": anomalies}

    time_domain = get_time_domain_features(nni, bpm_list)
    full_analysis = {'ecg_parameters': {"bpm_mean": bpm, "bpm_list": finalBpmList, "hrv_mean": hrv, "hrv_list": finalHrvList}, 'time_domain': time_domain,
                     'anomalies': anomalies, 'peaks': peaks, 'nni': nni}
    
    return full_analysis


def calculateIntervalParameters(data, fs):
    hrvList = []
    bpmList = []
    count = 0
    i = 1
    while i < len(data):
        if i % 1250 == 0:
            count += 1
            peaks, thrheshold_list = ACSPeakDetector3(data[i-1250:i], fs)
            nni = calculateNNI(peaks)
            hrv = calculateRmssd(nni)
            bpm, bpm_list = calculateBpm(nni)
            if bpm != 0:
                hrvList.append(hrv)
                bpmList.append(bpm)
        i+=1

    final_hrv = []
    for i in range(1, len(hrvList)):
        if i % 30 == 0:
            hrvMean = np.mean(hrvList[i-30:i])
            final_hrv.append(hrvMean)

    return bpmList, final_hrv


def calculateBpmFromRawData(ecg, fs):

    if (len(ecg) < fs*5):
        return "Error: data lenght too short"

    signal_average_mean_removed = movingAverageMean(ecg, fs)
    finalEcgArtifactRemoved = artifactRemoval(signal_average_mean_removed)

    newEcgFilt = bandpassFilt(finalEcgArtifactRemoved, 4, fs, 15, 5)
    derivateSignal = derivateStep(newEcgFilt, fs)
    squaredEcgfromderivate = np.power(np.abs(derivateSignal), 2)
    panTompkinsEcgfromderivate = movingAverageMeanPamTompkins(
        squaredEcgfromderivate, fs)

    peaks, thrheshold_list = ACSPeakDetector3(
        panTompkinsEcgfromderivate, fs)

    if len(peaks) < 4:
        return "Error: Not enough peaks found"

    nni = calculateNNI(peaks)

    if len(nni) < 4:
        return "Error: Not enough heart beat found"

    hrv = calculateRmssd(nni)
    bpm, bpm_list = calculateBpm(nni)

    parameters = {"bpm_mean": bpm, "bpm_list": bpm_list, "hrv_mean": hrv}

    return parameters


def calculateAnomalies(ecg, fs):
    try:
        signal_average_mean_removed = movingAverageMean(ecg, fs)
        finalEcgArtifactRemoved = artifactRemoval(signal_average_mean_removed)
        newEcgFilt = bandpassFilt(finalEcgArtifactRemoved, 4, fs, 15, 5)
        derivateSignal = derivateStep(newEcgFilt, fs)
        squaredEcgfromderivate = np.power(np.abs(derivateSignal), 2)
        panTompkinsEcgfromderivate = movingAverageMeanPamTompkins(
            squaredEcgfromderivate, fs)
        peaks, thrheshold_list = ACSPeakDetector3(
            panTompkinsEcgfromderivate, fs)
        nni = calculateNNI(peaks)
        beats_annotation = []
        pulse = 0
        normal_rythm = True
        beatClassificationList, pulse = beatsClassification2(
            nni, peaks, pulse, normal_rythm)
        for x in beatClassificationList:
            beats_annotation.append(x)
        # detect_afib(nni, fs)

        anomalies, anomalies_time, anomalies_indexing = countAnomalies(
            beats_annotation, peaks)

        parameters = {"anomalies": anomalies, "anomalies_time": anomalies_time,
                      "anomalies_indexing": anomalies_indexing}

        return parameters
    except:
        return "Error in signal, could not process the ECG Anomalies Search"


def getNNIFromSample(ecg, fs):
    signal_average_mean_removed = movingAverageMean(ecg, fs)
    finalEcgArtifactRemoved = artifactRemoval(signal_average_mean_removed)
    newEcgFilt = bandpassFilt(finalEcgArtifactRemoved, 4, fs, 15, 5)
    derivateSignal = derivateStep(newEcgFilt, fs)
    squaredEcgfromderivate = np.power(np.abs(derivateSignal), 2)
    panTompkinsEcgfromderivate = movingAverageMeanPamTompkins(
        squaredEcgfromderivate, fs)

    peaks, thrheshold_list = ACSPeakDetector3(panTompkinsEcgfromderivate, fs)

    if len(peaks) < 4:
        return "Not enough peaks found for the analysis."

    nni = calculateNNI(peaks)

    return nni

def getPeaksFromSample(ecg, fs):
    try:
        signal_average_mean_removed = movingAverageMean(ecg, fs)
        finalEcgArtifactRemoved = artifactRemoval(signal_average_mean_removed)
        newEcgFilt = bandpassFilt(finalEcgArtifactRemoved, 4, fs, 15, 5)
        derivateSignal = derivateStep(newEcgFilt, fs)
        squaredEcgfromderivate = np.power(np.abs(derivateSignal), 2)
        panTompkinsEcgfromderivate = movingAverageMeanPamTompkins(
            squaredEcgfromderivate, fs)

        peaks, thrheshold_list = ACSPeakDetector3(
            panTompkinsEcgfromderivate, fs)

        if len(peaks) < 2:
            return "Not enough peaks found."

        return peaks
    except:
        return "Signal error, could not be processed"

def run_new_ecg_analysis(file):
    """
    Run a full ECG analysis.

    * ARGS:
    - file: path of the raw ecg;
    - fs: the signal frequency sampling  // to be implemented;

    * Returns:
    - ecg: the raw ecg as list;
    - signal_average_mean_removed: the mean removed ECG as a list;
    - finalEcgArtifactRemoved: the final filtered ECG as a list;
    - parameter: a dictionary containing:
                    - bpm: the mean bpm;
                    - bpm_list: the intervalled bpm calculated every 5 seconds;
                    - hrv_mean: the heart rate variability mean;
                    - hrv_list: the intervalled hrv calculated every 5 seconds;
                    - anomalies: the 5 categories anomalies (normal, extrabeat, pvc, escape, fibrillation beats);
    - anomalies_time: 
    - anomalies_indexing:
    - time_domain:
    - peaks:
    - corr_peaks:
    - beatClassificationList:
    """
    ecg = np.genfromtxt(file, delimiter=',')[:500000]
    fs = 250

    signal_average_mean_removed = []
    finalEcgArtifactRemoved = []

    time_domain = {}
    frequencies = {}
    parameter = {}
    
    anomalies_time = {}
    time_domain = {} 
    anomalies_indexing = {}
    peaks = []
    corr_peaks = []
    beatClassificationList = []
    
    percentage = calculatePercentageFromData(ecg, fs)

    signal_average_mean_removed = movingAverageMean(ecg, 5)
    finalEcgArtifactRemoved = artifactRemoval(signal_average_mean_removed, fs)
    newEcgFilt = bandpassFilt(finalEcgArtifactRemoved, 4, fs, 15, 5)
    derivateSignal = derivateStep(newEcgFilt, fs)
    squaredEcgfromderivate = np.power(np.abs(derivateSignal), 2)
    panTompkinsEcgfromderivate = movingAverageMeanPamTompkins(
        squaredEcgfromderivate, fs)
    
    peaks, thrheshold_list = ACSPeakDetector3(panTompkinsEcgfromderivate, fs)
    corr_peaks = []

    # EXTRACT BREATH FROM ECG
    newList=[]
    y = 0
    z = 0
    for i in peaks:
        for x in range(z, i):
            newList.append(signal_average_mean_removed[i])
            z = i
        y += 1
    
    thrheshold_list = gaussian_filter1d(newList, sigma=75)
    thrheshold_list = [i for i in thrheshold_list]
    thrheshold_list = movingAverageMean(thrheshold_list, 750)
    brt_extracted = [i*200 for i in thrheshold_list]

    for i in peaks:
        a = np.array(finalEcgArtifactRemoved[i-30:i+30])
        idx = np.argmax(a)
        #idx = finalEcgArtifactRemoved.index(correct_peak)
        corr_peaks.append(i-(30-idx))

    nni = calculateNNI(corr_peaks)
    hrv = calculateRmssd2(nni)

    bpm, bpm_list = calculateBpm(nni)

    finalBpmList, finalHrvList = calculateIntervalParameters(
        panTompkinsEcgfromderivate, fs)

    beats_annotation = []
    pulse = 0
    i = 0
    beatClassificationList = new_tsipouras_2(signal_average_mean_removed, nni, corr_peaks, pulse, fs)
    
    anomalies, anomalies_time, anomalies_indexing = countAnomalies3(beats_annotation, corr_peaks, fs)
    beat_list = countAnomalies(beatClassificationList, corr_peaks)

    parameter = {"bpm_mean": bpm, "bpm_list": finalBpmList,
                        "hrv_mean": hrv, "hrv_list": finalHrvList, "anomalies": anomalies}
    
    if len(nni)>0:
        time_domain = get_time_domain_features(nni, bpm_list)
        frequencies = get_hrv_frequency_measures(nni)

        # print(beatClassificationList)
        # print(frequencies)
        # print(time_domain)
        # time_domain = {}
    
    return ecg, signal_average_mean_removed, finalEcgArtifactRemoved, parameter, anomalies_time, time_domain, anomalies_indexing, peaks, brt_extracted, corr_peaks, beatClassificationList


def run_stress_analysis(path, fs):
        
    fileList = []

    for filename in os.listdir(path):
        print(filename)
        fileList.append(path+filename)

    fileList.sort()

    hf_list = []
    lf_list = []
    lf_hf_ratio_list = []
    bpm_mean_list = []

    for f in fileList:
        ecg = np.genfromtxt(f, delimiter=',')
        fs = 250
        i = 1
        while i < len(ecg):
            if i % 45000 == 0:
                signal_average_mean_removed = movingAverageMean(ecg[i-45000:i], 5)
                finalEcgArtifactRemoved = artifactRemoval(signal_average_mean_removed, fs)
                newEcgFilt = bandpassFilt(finalEcgArtifactRemoved, 4, fs, 15, 5)
                derivateSignal = derivateStep(newEcgFilt, fs)
                squaredEcgfromderivate = np.power(np.abs(derivateSignal), 2)
                panTompkinsEcgfromderivate = movingAverageMeanPamTompkins(
                    squaredEcgfromderivate, fs)
                peaks, thrheshold_list = ACSPeakDetector3(panTompkinsEcgfromderivate, fs)
                corr_peaks = []

                """ plt.plot(finalEcgArtifactRemoved)
                plt.show() """

                for p in peaks:
                    a = np.array(finalEcgArtifactRemoved[p-30:p+30])
                    idx = np.argmax(a)
                    #idx = finalEcgArtifactRemoved.index(correct_peak)
                    corr_peaks.append(p-(30-idx))

                nni = calculateNNI(corr_peaks)
                bpm, bpm_list = calculateBpm(nni)
                if len(nni)>0:
                    time_domain = get_time_domain_features(nni, bpm_list)
                    frequencies = get_hrv_frequency_measures(nni)
                
                    hf_list.append(frequencies['hf']/1000)
                    lf_list.append(frequencies['lf']/1000)
                    lf_hf_ratio_list.append(frequencies['lf_hf_ratio'])
                    bpm_mean_list.append(bpm)
                
                # time_domain = {}
            i=i+1
                                                                            
    plt.plot(hf_list, label="HF Trend 3 minutes")
    plt.plot(lf_list, label="LF Trend 3 minutes")
    plt.plot(lf_hf_ratio_list, label="HF/LF ratio Trend 3 minutes")
    # plt.plot(bpm_mean_list, label="HR Trend 3 minutes")
    plt.legend()
    plt.show()
    
    return