"""
Script that process a given interval.
It is suggested to use 10 seconds of data,
less timing could interfere on the peak detection
"""

from ACS_ecg_analysis.ecg_processing.ar_detect import *

arrhytmhias_list = []
beat_details_array = []

def process_signal(signal, bpm, rpeaks, nni, sampling_rate, pulse, beat_list, consecutive_spv_hr, consecutive_v_hr, last_nni):
    """
    Process the signal and finds anomalies every given index,
    it is suggested to use it in a while loop of 10 sec of signal, 
    to compute the time divide the signal index for the sampling rate.
    **ARGS
    - signal: ecg signal
    - bpm: this interval heart rate
    - rpeaks: list of qrs detected
    - nni: list of normal beat to beat (RR)
    - sampling_rate: signal sampling rate
    - pulse: number of beats as atrial fibrillation
    - beat_list: list of beat type 
    - consecutive_spv_hr: number of beats as supraventricular tachycardia
    - consecutive_v_hr: number of beats as ventricular tachycardia
    ** RETURNS
    - arrhytmhias_list: list of rhythm every analyzed time interval,
    - beat_list: list of beats types
    - pulse = necessary to detect Atrial Fibrillation
    - consecutive_spv_hr = necessary to find Supraventricular Tachycardia
    - consecutive_v_hr =  necessary to detect Ventricular Tachycardia
    - last_nni = it save the last nni and add it to the next check nni list to have a full analysis
    """

    print("ENTERED ADVANCED SIGNAL PROCESSING")
    beat_list = []
    arrhytmhias_list = []
    beat_details_array = []

    # Set sinus rythm as true every time the function is called
    sinus_rythm = True

    # Set rythm as normal every time the function is called
    normal_rythm = True

    # Set the beat as normal every time the function is called
    beat_type = 'Normal'

    # Set the beat rythm as normal every time the function is called
    beat_rythm = 'Normal'

    # every 2500 at a sampling rate of 250Hz it starts the analysis
    # Get the actual dataset to prevent analyzing a big amount of data and speed the process
    try:
        
        # Add previous last nni to the start of the actual nni list if present
        if len(last_nni) > 0:
            nni = np.concatenate((last_nni, nni))
            last_nni = []

        last_nni.append(nni[-1])
        last_nni = np.array(last_nni)

        beat_list, beat_type, sinus_rythm = analyze_beats(signal, sampling_rate, beat_type, beat_list, sinus_rythm, rpeaks, nni)
        
        # Get the qrs mean of the analyzed interval
        qrs_mean = get_qrs_mean(signal, rpeaks, sampling_rate)

        # Check for Atrial Fibrillation
        if beat_rythm != 'Ventricular Tachycardia':
            pulse, fibrillation_beat, normal_rythm, beat_rythm = detect_fibrillation(signal, sampling_rate, pulse, normal_rythm, beat_rythm, nni)

        # Check for Junctional Tachycardia - Not actually working
        
        # junctional_tachycardia(signal, sampling_rate)

        # Check for Supraventricular Tachycardia if the rythm is normal
        """ if fibrillation_beat == True and bpm > 120:
            beat_rythm, pulse = supraventricular_tachycardia(signal, sampling_rate, pulse, beat_rythm, rpeaks=rpeaks, nni=nni)
        """

        # TODO = implement the new alghorythm for TV in function
        """ # Check for Ventricular Tachycardia
        if qrs_mean > 120 and bpm > 120:
            beat_rythm, beat_type, pulse = ventricular_tachycardia(
                signal, sampling_rate, pulse, beat_rythm)
            if beat_type != 'Normal':
                beat_list.append(beat_type) """

        beat_rythm = ventricular_tachy_detect(signal, sampling_rate, beat_rythm, rpeaks = rpeaks, nni = nni)

        # Check for Bradycardia
        beat_rythm = detect_bradycardia(signal, sampling_rate, sinus_rythm, beat_rythm, bpm=bpm)

        # Check for Tachycardia
        beat_rythm = detect_tachycardia(signal, sampling_rate, beat_rythm, bpm=bpm)
    except:
        pass

    arrhytmhias_list.append(beat_rythm)
    for beats in beat_list:
        beat_details_array.append(beats)

    print(arrhytmhias_list)
    print(beat_details_array)

    return arrhytmhias_list, beat_details_array, pulse, consecutive_spv_hr, consecutive_v_hr, last_nni

