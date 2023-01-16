"""
Function to detect arrythmia using tsipouras et al. algorithm
https://www.researchgate.net/publication/3998050_Arrhythmia_classification_using_the_RR-interval_duration_signal
In addition use the Neurokit2 PQRST Complex analysis to define beats and rythm
Need to do more tests to verify:
- junctional_tachycardia()
"""

import time
from ACS_ecg_analysis.ecg_processing.tools.utils import *

# Set Parameters following Tsipouras study
a = 0.9
b = 0.9
c = 1.5


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


def detect_fibrillation(nni, sampling_rate, pulse, normal_rythm, beat_rythm, fibrillation_beat=False):
    """
    Detect Fibrillation rythm - to use in a loop while iterating over the signal data
    ** Arguments:
    - nni: list of normal beat to beat (RR)
    - sampling_rate: sampling frequency of the sample
    - pulse: beat as fibrillation detected
    - fibrillation_beat: define if the beat is fibrillation
    - normal_rythm: define the type of the rythm
    - beat_rythm: define the interval rythm
    """

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


def detect_bradycardia(sampling_rate, sinus_rythm, beat_rythm, bpm):
    """
    Set ad default < 50bpm
    **ARGS
    - sampling_rate: the sampling frequency
    - sinus_rythm: if the rhytm is sinus of not
    - beat_rythm: the beat rythm
    - bpm: the actual heart rate
    RETURN
    - beat_rythm: the beat rythm
    """

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


def detect_tachycardia(sampling_rate, beat_rythm, bpm):
    """
    Define in Bradycardic Rythm
    Set ad default < 50bpm
    **ARGS
    - sampling_rate: the sampling frequency
    - beat_rythm: the beat rythm
    - bpm: the actual heart rate
    RETURN
    - beat_rythm: the beat rythm
    """

    if bpm > 90 and beat_rythm == 'Normal':
        beat_rythm = 'Sinus Tachycardia'
        return beat_rythm
    else:
        return beat_rythm


def ventricular_tachycardia(nni, sampling_rate, pulse, beat_rythm):
    # print('VENTR TACHY CHECK')
    "QRS > 120, hr > 120"
    x = 0
    beat_type = 'Normal'
    consecutive_v_hr = 0
    total_v_hr = 0
    # VT has a faster HR, so we need to scale the data at faster sampling rate

    hr_list = []

    while x < len(nni)-1:
        x += 1
        hr = (60/(nni[x]))*1000
        # print(hr)
        if hr > 110:
            hr_list.append(hr)
            consecutive_v_hr = sum_el(consecutive_v_hr)
            if consecutive_v_hr == 1:
                index_started = x
            if x == len(nni)-1:
                total_v_hr = consecutive_v_hr
        else:
            total_v_hr = consecutive_v_hr
            total_v_hr = consecutive_v_hr
            consecutive_v_hr = 0
            index_started = 0

        if total_v_hr > 3:
            mean_hr = np.mean(hr_list)
            if mean_hr > 250:
                beat_rythm = 'Ventricular Flutter'
                pulse = 0
            else:
                beat_rythm = 'Ventricular Tachycardia'
                pulse = 0

        # Test to detect couple or triplette from a ventricular series if not VT of VF
        if total_v_hr == 2:
            beat_type = 'Ventricular Couple'
            pulse = 0

        if total_v_hr == 3:
            beat_type = 'Ventricular Triplette'
            pulse = 0

    return beat_rythm, beat_type, pulse


def supraventricular_tachycardia(sampling_rate, pulse, beat_rythm, rpeaks, nni):
    """
    SUpraventricula Tachycardia : QRS < 120, hr > 140, pr < 120
    Junctional Tachycardia: hr > 100, qrs < 120,
    """
    # print('SUPRV CHECK')

    print('PASSING')
    x = 0
    consecutive_spv_hr = 0
    total_spv_hr = 0
    while x < len(nni)-1:
        hr = (60/(nni[x]))*1000
        sinus = check_for_p_wave(signal, x, rpeaks)
        pr = get_pr(rpeaks, x, signal, sampling_rate)
        if sinus:
            if pr < 50:
                if hr > 140:
                    consecutive_spv_hr = sum_el(consecutive_spv_hr)
                    if x == len(nni)-1:
                        total_spv_hr = consecutive_spv_hr
                else:
                    total_spv_hr = consecutive_spv_hr
                    consecutive_spv_hr = 0
                    time_start = 0

                if total_spv_hr > 3:
                    beat_rythm = 'Supraventricular Tachycardia'
                    pulse = 0

                if total_spv_hr == 2:
                    beat_rythm = 'Supraventricular Couple'
                    pulse = 0

                if total_spv_hr == 3:
                    beat_rythm = 'Supraventricular Triplette'
                    pulse = 0

        x += 1

    return beat_rythm, pulse


def junctional_tachycardia(rpeaks, nni, signal, sampling_rate):
    """
    Juncitonal Tachycardia: qrs < 120, hr > 100, pr < 120
    """

    pr = pr_mean(rpeaks, signal, sampling_rate)
    hr = get_mean_hr(signal, sampling_rate)

    if hr > 100:
        if pr < 120:
            beat_rythm = 'Junctional Tachycardia'

            return beat_rythm


def premature_beats(signal, nni, x, rpeaks, sinus_rythm, sampling_rate):
    """
    Atrial, nodal and supraventricular premature beats
    Checks for Interval anomalies and analyze the beat type to distinguish the Ventricula beat.
    If the pause after a premature beat is complete then it is an atrial premature beat, if
    it is incomplete then it is a ventricular beat.
    """
    beat_type = 'Normal'
    if x < len(nni)-1:
        if nni[x] < a*nni[x-1] and nni[x-1] < b*nni[x+1]:

            if nni[x] + nni[x+1] < 2 * nni[x-1]:

                qrs = get_qrs(rpeaks, x, signal, sampling_rate)

                if qrs > 100:
                    # Noticed that there are some PVCs seen as atrial, this should fix, need a review
                    beat_type = 'Ventricular Premature Beat'
                    this_rpeak = rpeaks[x]
                else:
                    beat_type = 'Atrial Premature Beat'
                    this_rpeak = rpeaks[x]
            else:
                this_rpeak = rpeaks[x]
                beat_type = 'Ventricular Premature Beats'

    return beat_type


def escape_beat(signal, x, nni, rpeaks, sampling_rate):
    '''
    Check for Escape beats
    If the qrs > 120 the is a ventricular beat
    '''
    beat_type = 'Normal'
    if x < len(nni):
        if nni[x-1] > c*nni[x]:
            qrs = get_qrs(rpeaks, x, signal, sampling_rate)
            if qrs > 100:
                this_rpeak = rpeaks[x]
                beat_type = 'Ventricular Escape beat'
            else:
                this_rpeak = rpeaks[x]
                beat_type = 'Atrial Escape beat'

    return beat_type


def sinus(signal, nni, sampling_rate):
    '''
    Check for sinus rythm - need fixes if need to be used
    '''

    sinus = True
    x = 0
    sinus_list = []

    while x < len(nni)-1:
        x += 1

        _, waves_peak = nk.ecg_delineate(signal, rpeaks)

        # Exclude the P peak if on same position of previous T Peak
        if x > 1:
            try:
                previous_t_peak = waves_peak['ECG_T_Peaks'][x-2:x][0]
                actual_p_peak = waves_peak['ECG_P_Peaks'][x-1:x][0]
                if actual_p_peak in range(previous_t_peak-20, previous_t_peak+20):
                    waves_peak['ECG_P_Peaks'][x-1:x] = [float("nan")]
            except:
                pass

        p_wave = waves_peak['ECG_P_Peaks'][x-1:x]

        if np.isnan(p_wave):
            sinus = False

        sinus_list.append(sinus)

    return sinus_list


def analyze_beats(signal, sampling_rate, beat_type, beat_list, sinus_rythm, rpeaks, nni):
    """
    Function that analyze all the beats detected into the given sample interval and gives an
    array as output with the beats detected in that interval, the interval rythm and check if the
    interval is sinus or not.
    **ARGS
    - signal: sample to analyze
    - sampling_rate: sample recording frequency
    - nni: list of beat intervals
    - rpeaks: list of peaks
    - beat_type: the beat type
    - beat_list: the beats list
    - sinus_rythm: the type of the rythm
    **RETURNS
    - beat_type: the last beat type
    - beat_list: the list of the beat type
    - sinus_rythm: if the interval rythm is sinus or not
    """

    count_vent_beats = 0

    x = 0

    while x < len(nni):
        # Check for escape beats
        beat_type = escape_beat(
            signal, x, nni, rpeaks, sampling_rate)

        if beat_type != 'Normal':
            beat_list.append(beat_type)

        if beat_type != 'Ventricular Escape beat' and beat_type != 'Atrial Escape beat':
            # Check for Premature beats
            beat_type = premature_beats(
                signal, nni, x, rpeaks, sinus_rythm, sampling_rate)

            if beat_type != 'Normal':
                beat_list.append(beat_type)

        # Analize every beat of the interval selected
        _, waves_peak = nk.ecg_delineate(
            signal, rpeaks, sampling_rate)

        # Visualize the T-peaks, P-peaks, Q-peaks and S-peaks
        # Exclude the P peak if on same position of previous T Peak
        if x > 1:
            try:
                previous_t_peak = waves_peak['ECG_T_Peaks'][x-2:x][0]
                actual_p_peak = waves_peak['ECG_P_Peaks'][x-1:x][0]

                if actual_p_peak in range(previous_t_peak-10, previous_t_peak+10):
                    waves_peak['ECG_P_Peaks'][x-1:x] = [float("nan")]
                    sinus_rythm = False
            except:
                pass

        # Find pause
        if x < len(nni):
            if nni[x-1] > 1600:
                beat_type = 'Pause'
                beat_list.append(beat_type)

        if beat_type == 'Normal':
            beat_list.append(beat_type)

        x += 1

    return beat_list, beat_type, sinus_rythm


def ventricular_tachy_detect(signal, sampling_rate, beat_rythm, rpeaks, nni):
    """
    Detect Ventricular Tachycardia. It checks if there are more than 3 Ventricular beats in a raw.
    It can work with given rpeaks and nni or without, if are not presents then will calculate them.
    ** ARGS:
    - signal = ecg signal
    - sampling_rate = record frequency of the ecg
    - beat_rythm = the beat rythm
    - rpeaks = rpeaks array
    - nni = nni array
    ** RETURNS:
    - beat_rythm = the beat rythm af the interval
    """
    
    count_vent_beats = 0
    x = 0
    while x < len(nni):
        qrs = get_qrs(rpeaks, x, signal, sampling_rate)
        if qrs > 100:
            beat_type = 'Ventricular Beat'
            count_vent_beats += 1

            if count_vent_beats >= 3:
                hr = get_mean_hr(signal, sampling_rate, nni[x-3:x])
                if hr > 90:
                    beat_rythm = 'Ventricular Tachycardia'
            else:
                count_vent_beats = 0 
        x += 1

    return 