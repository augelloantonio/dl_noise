import numpy as np
import datetime
from math import isclose
from ACS_ecg_analysis.ecg_processing.signal_analysis import calculateBpm, calculateNNI
from ACS_ecg_analysis.ecg_processing.signal_peak_detector import ACSPeakDetector3
from ACS_ecg_analysis.ecg_processing.utils import mean

def beatsClassification(nni, rpeaks, pulse):
    ecgAnomaliesList = []
    a = 0.9
    b = 0.9
    c = 1.5
    pulse = 0

    beat_list_peaks = []
    nni_list = []

    for i in nni:
        nni_list.append(i)
    nni_list.append(0)
    nni_list.append(0)

    for i in range(0, len(nni_list)-2):

        ecgAnomaliesList.append("n")

        # CAT 5 - VF/VT
        if nni_list[i+1] < 600 and nni_list[i+2] < nni_list[i+1] < 800:
            if (nni_list[i] + nni_list[i+1] + nni_list[i+2] < 1800) or (nni_list[i] < 800 and nni_list[i+1] < 800 and nni_list[i+2] < 800):
                pulse += 1

            if (pulse > 4):
                ecgAnomaliesList[-1] = "vf/vt"
                beat_list_peaks.append(rpeaks[i % 3])
                i = i-pulse

        if ecgAnomaliesList[i-1] == "n":
            # CAT 3 & CAT 2
            if nni_list[i+1] < nni_list[i]*a and nni_list[i] < b*nni_list[i+2]:
                if nni_list[i+1]+nni_list[i+2] < 2*nni_list[i]:
                    # CAT 2
                    ecgAnomaliesList[-1] = "atrial/nodal/supraventricular beat"
                    beat_list_peaks.append(rpeaks[i % 3])
                else:
                    # CAT 3
                    ecgAnomaliesList[-1] = "ventricular premature beats"
                    beat_list_peaks.append(rpeaks[i % 3])

        if ecgAnomaliesList[i-1] == "n":
            # CAT 4
            if nni_list[i+1] > c*nni_list[i]:
                ecgAnomaliesList[-1] = "escape beat"
                beat_list_peaks.append(rpeaks[i % 3])

    return ecgAnomaliesList


def beatsClassification2(nni, rpeaks, pulse, normal_rythm):

    ecgAnomaliesList = []
    a = 0.5
    b = 0.5
    c = 2.0
    pulse = 0

    beat_list_peaks = []

    for i in range(0, len(nni) % 2):
        nni.append(0)

    for i in range(0, len(nni)-2):
        ecgAnomaliesList.append("n")

        # CAT 5 - VF/VT
        if nni[i+1] < 600 and nni[i+2] < nni[i+1] < 800:
            if (nni[i] + nni[i+1] + nni[i+2] < 1800) or (nni[i] < 800 and nni[i+1] < 800 and nni[i+2] < 800):
                pulse += 1
                # if not normal_rythm:
                #pulse += 1

            if (pulse > 4):
                ecgAnomaliesList[-1] = "vf/vt"
                beat_list_peaks.append(rpeaks[i % 3])
                i = i-pulse

        if ecgAnomaliesList[i-1] == "n":
            # CAT 3 & CAT 2
            if nni[i+1] < nni[i]*a and nni[i] < b*nni[i+2]:
                if nni[i+1]+nni[i+2] < 2*nni[i]:
                    # CAT 2
                    ecgAnomaliesList[-1] = "atrial/nodal/supraventricular beat"
                    beat_list_peaks.append(rpeaks[i % 3])
                else:
                    # CAT 3
                    ecgAnomaliesList[-1] = "ventricular premature beats"
                    beat_list_peaks.append(rpeaks[i % 3])

        if ecgAnomaliesList[i-1] == "n":
            # CAT 4
            if nni[i+1] > c*nni[i]:
                ecgAnomaliesList[-1] = "escape beat"
                beat_list_peaks.append(rpeaks[i % 3])

    return ecgAnomaliesList, pulse

def new_tsipouras_2(signal, nni, rpeaks, pulse, fs):
    beatClassificationList = []

    a = 0.9
    b = 0.9
    c = 1.5

    for i in rpeaks:
        beatClassificationList.append("N")
    x = 0

    pulse = 0
    threshold = 0
    beat_list_peaks = []
    start_vf_loop = False
    counter_anomalies_loop = 0
    peak_list = []
    
    for i in range(0, len(nni)-2):
        try:
            # the i iteration of the article = x 
            this_nni = [nni[i], nni[i+1], nni[i+2]]
            this_rr = "cat_2"
            vf_nni = []

            ecgANList = ["N", "N", "N"]
            hearth_rythm = "normal"

            if nni[i+1] < 600 and nni[i+1]*1.8>nni[i]:
                start_vf_loop = True
                    
            if start_vf_loop:
                x = i
                while start_vf_loop:
                    if nni[x+1] < 600 and nni[x+1]*1.8>nni[x]:
                        # beatClassificationList[x+1] = "atrial/nodal/supraventricular beat"
                        vf_nni = [nni[x], nni[x+1], nni[x+2]]
                        
                        # Starts of VF (fribrillation/flutter)
                        if nni[x] < 700 and nni[x+1] < 700 and nni[x+2]<700 or (nni[x] + nni[x+1] + nni[x+2]) < 1700: #c2 
                            hearth_rythm = "VF"
                            pulse = pulse+1
                            # beatClassificationList[x+1] = "atrial/nodal/supraventricular beat"
                        else:
                            if pulse>=4:
                                # print("FIBRILLATION")
                                start_fibrillation = datetime.timedelta(seconds=rpeaks[i+1]/fs)
                                end_fibrillation = datetime.timedelta(seconds=rpeaks[x+1]/fs)
                                # print("VF Start: " + str(start_fibrillation) + " - VF Stop: " + str(end_fibrillation))
                            beatClassificationList[x+1] = "N"
                            pulse = 0
                        #  x = x+3 # loop every 3 beats (sequencial 3 nni windows)
                        x = x+1 # loop every 3 beats (moving windows nni)
                    else:
                        start_vf_loop = False
                        i=x-pulse 
                        pulse = 0
                
            else:
                """ C3 or C4 or C5 """    
                if nni[i+1]*1.2 < nni[i] and nni[i+1]*1.2 < nni[i+2]: #C3
                    # Isolated PVC
                    beatClassificationList[i+2] = "atrial/nodal/supraventricular beat" # CATEGORY 2
                    ecgANList[1] = "atrial/nodal/supraventricular beat" # CATEGORY 2
                    beat_list_peaks.append(rpeaks[i+2])

                """ if abs(nni[i]-nni[i+1])<300 and nni[i]<700 or nni[i+1]<700 and nni[i+2]>1.2*mean(nni[i], nni[i+1]): #C4
                    # PVC Couplets
                    is_equal_1 = isclose(nni[i+1], nni[i], abs_tol=1e-8)
                    if is_equal_1:
                        if nni[i+2] > nni[i+1]:
                            beatClassificationList[i+2] = "ventricular premature beats " # CATEGORY 3
                            beatClassificationList[i+1] = "ventricular premature beats" # CATEGORY 3
                            ecgANList[1] = "ventricular premature beats" # CATEGORY 3
                            ecgANList[0] = "ventricular premature beats" # CATEGORY 3
                            beat_list_peaks.append(rpeaks[i+2])

                if abs(nni[i+1]-nni[i+2])<300 and nni[i+1]<700 or nni[i+2]<700 and nni[i]>1.2*mean(nni[i+1], nni[i+2]): #C5 
                    # PVC Couplet
                    is_equal_2 = isclose(nni[i+1], nni[i+2], abs_tol=1e-8)
                    if is_equal_2:
                        if nni[i] > nni[i+1]:
                            beatClassificationList[i+2] = "ventricular premature beats" # CATEGORY 3
                            beatClassificationList[i+1] = "ventricular premature beats" # CATEGORY 3
                            ecgANList[1] = "ventricular premature beats" # CATEGORY 3
                            ecgANList[2] = "ventricular premature beats" # CATEGORY 3
                            beat_list_peaks.append(rpeaks[i+2]) """

                """
                PVC BEAT
                """               
                if threshold!=0:
                    if signal[rpeaks[i]][0] < threshold*0.2 or signal[rpeaks[i][0]] > threshold*1.8:
                    # if signal[rpeaks[i]][0] < threshold*0.3 and nni[i]+nni[i+2] >= 2*nni[i+1]:
                        beatClassificationList[i] = "ventricular premature beats" # CATEGORY 3
                        ecgANList[1] = "ventricular premature beats" # CATEGORY 3 

                if ecgANList[1] != "N":
                    if threshold!=0:
                        if signal[rpeaks[i][0]] < threshold*0.2 or signal[rpeaks[i][0]] > threshold*1.8:
                            beatClassificationList[i] = "ventricular premature beats" # CATEGORY 3
                            ecgANList[1] = "ventricular premature beats" # CATEGORY 3  """

                """ if threshold!=0:
                    if signal[rpeaks[i]][0] < threshold*0.4:
                        beatClassificationList[i] = "ventricular premature beats" # CATEGORY 3
                        ecgANList[1] = "ventricular premature beats" # CATEGORY 3 """

                try:
                    if beatClassificationList[i] == "N":
                        peak_list.append(signal[rpeaks[i+2]])
                        threshold = np.mean(peak_list)
                except:
                    pass
        except:
            pass

    i = 1
    while i<len(beatClassificationList)-2:
        if beatClassificationList[i] == "ventricular premature beats" and beatClassificationList[i-1]=="atrial/nodal/supraventricular beat":
            beatClassificationList[i-1] = "N"
        if beatClassificationList[i] == "ventricular premature beats" and beatClassificationList[i+1]=="atrial/nodal/supraventricular beat":
            beatClassificationList[i-1] = "N"
        i=i+1
    return beatClassificationList


def countAnomalies3(beatClassificationList, peaks, fs):
    count_n = 0
    count_pvc = 0
    count_escape_beats = 0
    count_fib = 0
    count_extrabeat = 0
    count_anomalies = 0
    count_tot_anomalies = 0
    
    anomalies_list_timing = {}
    anomalies_indexing = {}

    for index, val in enumerate(beatClassificationList):
        if index < len(beatClassificationList)-1:
            if val == "N":
                count_n += 1
                time = str(datetime.timedelta(seconds=peaks[index]/fs))
                anomalies_indexing[index]={"cat":"cat_1", "index":peaks[index]}
                anomalies_list_timing[index] = {"event":"N", "time":time}
            if val == "escape beat":
                anomalies_indexing[index]={"cat":"cat_4", "index":peaks[index]}
                count_tot_anomalies +=1
                count_escape_beats += 1
                count_anomalies +=1
                time = str(datetime.timedelta(seconds=peaks[index]/fs))
                anomalies_list_timing[index] = {"event":"escape beat", "time":time}
            if val == "ventricular premature beats":
                count_pvc += 1
                count_anomalies +=1
                time = str(datetime.timedelta(seconds=peaks[index]/fs))
                anomalies_list_timing[index] = {"event":"ventricular premature beats", "time":time}
                anomalies_indexing[index]={"cat":"cat_3", "index":peaks[index]}
                count_tot_anomalies +=1
            if val == "vf/vt":
                count_fib += 1
            if val == "atrial/nodal/supraventricular beat":
                count_extrabeat += 1
                count_anomalies +=1
                time = str(datetime.timedelta(seconds=peaks[index]/fs))
                anomalies_list_timing[index] = {"event":"atrial/nodal/supraventricular beat", "time":time}
                anomalies_indexing[index]={"cat":"cat_2", "index":peaks[index]}
                count_tot_anomalies +=1

    categories = {}
    categories[0] = {"cat_1":count_n, "cat_2":count_extrabeat, "cat_3":count_pvc, "cat_4":count_escape_beats, "cat_5":count_fib}

    return categories, anomalies_list_timing, anomalies_indexing

def detect_arrythmia(nni, normal_rythm):
    """
    If the difference between the max and the min nni
    is > 400 then there is an arrhythmia
    # Calculate Arrhythmia - Study:
    https://litfl.com/sinus-arrhythmia-ecg-library/#:~:text=The%20P%2DP%20interval%20varies%20widely,a%20variability%20of%20over%20400ms.
    """

    max_nni = np.max(nni)
    min_nni = np.min(nni)
    diff_nni = max_nni - min_nni

    beat_rythm = 'Normal'

    if diff_nni > 400:
        normal_rythm = False
        beat_rythm = 'Arrhythmia'

    return normal_rythm, beat_rythm


def detect_fibrillation(signal, sampling_rate, pulse, normal_rythm, beat_rythm, fibrillation_beat=False, nni=[]):
    """
    Detect Fibrillation rythm - to use in a loop while iterating over the signal data
    ** Arguments:
    - signal: ecg to process
    - sampling_rate: sampling frequency of the sample
    - pulse: beat as fibrillation detected
    - fibrillation_beat: define if the beat is fibrillation
    - normal_rythm: define the type of the rythm
    - beat_rythm: define the interval rythm
    """
    if nni is None:
        nni = calculateNNI(signal)

    x = 0

    # Check if the interval has a normal rythm
    normal_rythm, beat_rythm = detect_arrythmia(nni, normal_rythm)

    # Afib is an arithmic rythm

    if not normal_rythm:
        while x < len(nni):
            x += 1
            if x < len(nni)-2:
                # Detect VF if nni[i] < 600 msec
                if nni[x] < 600 and nni[x+1] < nni[x+2]:
                    pulse = sum_el(pulse)
                    fibrillation_beat = True

                    # if pulse is > 4 then it can be considered as category 5
                    if pulse > 4:
                        beat_rythm = 'Atrial Fibrillation'

    return pulse, fibrillation_beat, normal_rythm, beat_rythm


def detect_bradycardia(signal, sampling_rate, sinus_rythm, beat_rythm, bpm=None):
    """
    Set ad default < 50bpm
    **ARGS
    - signal: the ecg signal interval to process
    - sampling_rate: the sampling frequency
    - sinus_rythm: if the rhytm is sinus of not
    - beat_rythm: the beat rythm
    RETURN
    - beat_rythm: the beat rythm
    """

    if bpm == None:
        bpm = calculateBpm(signal)

    # Check for Bradycardia
    if bpm < 50:
        if not sinus_rythm:
            beat_rythm = 'Bradycardia'
            return beat_rythm
        else:
            beat_rythm = 'Sinus Bradycardia'
            """
            If the rythm is bradycardic and the P wave absent we could face a
            Third-degree, or complete, Seno-Atrial block
            """
            return beat_rythm
    else:
        return beat_rythm


def detect_tachycardia(signal, sampling_rate, beat_rythm, bpm=None):
    """
    Define in Bradycardic Rythm
    Set ad default < 50bpm
    **ARGS
    - signal: the ecg signal interval to process
    - sampling_rate: the sampling frequency
    - beat_rythm: the beat rythm
    RETURN
    - beat_rythm: the beat rythm
    """
    if bpm == None:
        bpm = get_mean_hr(signal, sampling_rate)

    if bpm > 90 and beat_rythm == 'Normal':
        beat_rythm = 'Sinus Tachycardia'
        return beat_rythm
    else:
        return beat_rythm


def detect_afib(signal, fs):
    
    ssdThreshold = 120
    prev_sdrr_true = False
    i = 1
    while i < len(signal):
        if i % fs*120 == 0:
            try:
                peaks = ACSPeakDetector3(signal[i-fs*120:i], fs)
                nni = calculateNNI(peaks)
                bpm = calculateBpm(nni)
                rrsd = np.std(nni)
                if bpm > 100:
                    if rrsd > ssdThreshold:
                        print("AFIB DETECTED")
                else:
                    if rrsd > ssdThreshold and prev_sdrr_true == False:
                        prev_sdrr_true = True
                    if rrsd > ssdThreshold and prev_sdrr_true == True:
                        if bpm > 100 and rrsd > ssdThreshold:
                            print("AFIB DETECTED")
                        else:
                            print("POTENTIAL AFIB DETECTED")
            except:
                pass
        i += 1


def countAnomalies(beatClassificationList, peaks):
    count_n = 0
    count_pvc = 0
    count_escape_beats = 0
    count_fib = 0
    count_extrabeat = 0
    count_anomalies = 0
    count_tot_anomalies = 0

    anomalies_list_timing = {}
    anomalies_indexing = {}
    categories = {}

    for index, val in enumerate(beatClassificationList):
        if index<=len(peaks):
            if val == "n":
                count_n += 1
            if val == "escape beat":
                print(index)
                anomalies_indexing[count_tot_anomalies] = {
                    "cat": "cat_4", "index": peaks[index]}
                count_tot_anomalies += 1
                count_escape_beats += 1
                count_anomalies += 1
                time = str(datetime.timedelta(seconds=peaks[index]/250))
                anomalies_list_timing[count_anomalies] = {
                    "event": "escape beat", "time": time}
            if val == "ventricular premature beats":
                count_pvc += 1
                count_anomalies += 1
                time = str(datetime.timedelta(seconds=peaks[index]/250))
                anomalies_list_timing[count_anomalies] = {
                    "event": "ventricular premature beats", "time": time}
                anomalies_indexing[count_tot_anomalies] = {
                    "cat": "cat_3", "index": peaks[index]}
                count_tot_anomalies += 1
            if val == "vf/vt":
                count_fib += 1
            if val == "atrial/nodal/supraventricular beat":
                count_extrabeat += 1
                count_anomalies += 1
                time = str(datetime.timedelta(seconds=peaks[index]/250))
                anomalies_list_timing[count_anomalies] = {
                    "event": "atrial/nodal/supraventricular beat", "time": time}
                anomalies_indexing[count_tot_anomalies] = {
                    "cat": "cat_2", "index": peaks[index]}
                count_tot_anomalies += 1

        categories = {"cat_1": count_n, "cat_2": count_extrabeat,
                    "cat_3": count_pvc, "cat_4": count_escape_beats, "cat_5": count_fib}

    return categories, anomalies_list_timing, anomalies_indexing
