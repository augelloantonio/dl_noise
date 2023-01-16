import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import style

from ACS_ecg_analysis.plot.plot_data import plotData
from ACS_ecg_analysis.ecg_processing.signal_anomalies_detector import new_tsipouras_2, countAnomalies3, beatsClassification, countAnomalies, beatsClassification2, detect_arrythmia, detect_afib
from ACS_ecg_analysis.ecg_processing.signal_filtering import movingAverageMean, artifactRemoval, bandpassFilt, derivateStep, movingAverageMeanPamTompkins
from ACS_ecg_analysis.ecg_processing.signal_peak_detector import ACSPeakDetector, panPeakDetect, newACSPeakDetector, newACSPeakDetector2, ACSPeakDetector3
from ACS_ecg_analysis.ecg_processing.signal_analysis import calculateNNI, calculateBpm, calculateRmssd
from ACS_ecg_analysis.mit_processing.mit_reader import getAnnotation, countAnnotationAnomalies
from ACS_ecg_analysis.mit_processing.load_annotation import loadAnnotationSample
from ACS_ecg_analysis.mit_processing.mit_analysis import checkNegative, checkPositive
from ACS_ecg_analysis.ecg_processing.signal_time_domain_analysis import get_time_domain_features
from ACS_ecg_analysis.ecg_processing.signal_frequencies_analysis import get_hrv_frequency_measures
from ACS_ecg_analysis.ecg_percentage import calculatePercentageFromData
from ACS_ecg_analysis.ecg_processing.processing import process_signal


def start_ecg_analysis(signal, sampling_rate, check_bpm_interval, interval_analysis):
    """
    Start Process function: it filter the signal and then process it
    **ARGS
    - signal: the raw ecg signal
    - sampling_rate: the sample sampling rate frequency in Hz
    - check_bpm_interval: the desidered bpm analysis interval (suggested not less that 750(3sec))
    **RETURNS
    - arrhytmhias_list: list of arrhytmhias detected
    - beat_list: list of 
    - hrv_measures: 
    N.B.:
    * seconds = n. of datas/sampling rate
    * The signal file should have a 'Data' column where are stored the ecg raw data
    """

    # Local const to call
    pulse = 0
    consecutive_spv_hr = 0
    consecutive_v_hr = 0
    last_nni = []
    beat_list = []
    arrhytmhias = []
    beat_details = []
    arrhytmhias_list = []
    beat_details_list = []
    bpm = 0
    col_lenght = 0

    print('START ANALYSIS SIGNAL')
    
    signal_average_mean_removed = movingAverageMean(signal, 5)
    finalEcgArtifactRemoved = artifactRemoval(signal_average_mean_removed, sampling_rate)
    newEcgFilt = bandpassFilt(finalEcgArtifactRemoved, 4, sampling_rate, 15, 5)
    derivateSignal = derivateStep(newEcgFilt, sampling_rate)
    squaredEcgfromderivate = np.power(np.abs(derivateSignal), 2)
    panTompkinsEcgfromderivate = movingAverageMeanPamTompkins(
        squaredEcgfromderivate, sampling_rate)
            
    size  = len(finalEcgArtifactRemoved)
    print(size)

    # loop over the signal to analyze it
    i = 0
    bpm = 0
    while i < size:
        i+=1
        # Get BPM
        if i % check_bpm_interval == 0 and np.mean(panTompkinsEcgfromderivate[i-check_bpm_interval:i]) != 0:
            peaks, thrheshold_list = ACSPeakDetector3(panTompkinsEcgfromderivate[i-check_bpm_interval:i], sampling_rate)
            corr_peaks = []

            for p in peaks:
                a = np.array(finalEcgArtifactRemoved[p-30:p+30])
                idx = np.argmax(a)
                #idx = finalEcgArtifactRemoved.index(correct_peak)
                corr_peaks.append(p-(30-idx))

            nni = calculateNNI(corr_peaks)
            hrv = calculateRmssd(nni)

            bpm, bpm_list = calculateBpm(nni)

            #print("bpm: " + str(bpm))
        
        if i % interval_analysis == 0 and bpm > 30 and not math.isnan(bpm):
            # Detect anomalies and make a list of them
            signal = finalEcgArtifactRemoved[i-interval_analysis:i]
            arrhytmhias, beat_details, pulse,consecutive_spv_hr, consecutive_v_hr, last_nni = process_signal(signal, bpm, corr_peaks, nni, sampling_rate, pulse, beat_list, consecutive_spv_hr, consecutive_v_hr, last_nni)
            print(arrhytmhias, beat_details)
    for x in arrhytmhias:
        arrhytmhias_list.append(x)

    for v in beat_details:
        beat_details_list.append(v)
    # arrhytmhias_list, beat_details_array, pulse,consecutive_spv_hr, consecutive_v_hr, last_nni = process_signal(signal, sampling_rate, pulse, beat_list, consecutive_spv_hr, consecutive_v_hr, last_nni)
    
    # hrv_measures = get_hrv_measures(signal_filtered, sampling_rate) 
    hrv_measures = {}
    return arrhytmhias_list, beat_details_list, hrv_measures