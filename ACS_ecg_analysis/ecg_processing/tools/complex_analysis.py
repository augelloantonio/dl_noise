import neurokit2 as nk
import numpy as np

def get_qtc(rpeaks, i, nni, signal, sampling_rate):
    """
    Get the qtc mean of the analyzed rr list
    ** Arguments
    * rpeaks: list of peaks
    * i: iteration counter to get exact qtc of that data
    * nni: list of rr intervals
    * signal: dataset to analyze
    """

    signal_dwt, waves_dwt = nk.ecg_delineate(
        signal, rpeaks, sampling_rate, method="dwt", show=False, show_type='all')
    signal_dwt, t_offset = nk.ecg_delineate(
        signal, rpeaks, sampling_rate,  show=False, show_type='peaks')['ECG_T_Offsets']

    qt = np.subtract(waves_dwt['t_offset'][i:i+1],
                     waves_dwt['ECG_R_Onsets'][i:i+1])

    nni_mean = np.mean(nni[i])

    qt_mean = np.nanmean(qt)
    qtc = (qt_mean/np.sqrt(nni_mean/1000))
    qtc = (qtc/sampling_rate)*1000

    return int(qtc)


def get_qtc_mean(rpeaks, nni, signal):
    """
    Get the qtc mean of the analyzed rr list
    * rpeaks: list of peaks
    * nni: list of rr intervals
    * signal: dataset to analyze
    """

    signal_dwt, waves_dwt = nk.ecg_delineate(
        signal, rpeaks, sampling_rate, method="dwt", show=False, show_type='all')
    signal_dwt, t_offset = nk.ecg_delineate(
        signal, rpeaks, sampling_rate,  show=False, show_type='peaks')['ECG_T_Offsets']

    qt = np.subtract(waves_dwt['t_offset'], waves_dwt['ECG_R_Onsets'])

    qt = np.divide(qt, sampling_rate)

    nni_mean = np.mean(nni)

    qt_mean = np.nanmean(qt)
    qtc = (qt_mean/np.sqrt(nni_mean/1000))
    qtc = (qtc/sampling_rate)*1000

    return qtc


def get_qrs(rpeaks, i, signal, sampling_rate):
    """
    Get the qrs mean of the analyzed rr list
    ** Arguments
    * rpeaks: list of peaks
    * i: iteration counter to get exact qtc of that qrs
    * signal: dataset to analyze
    * sampling_rate: sampling rate of the analyzed sample
    """

    try:
        _, waves_peak = nk.ecg_delineate(signal, rpeaks, sampling_rate=sampling_rate)

        qrs = np.subtract(waves_peak['ECG_S_Peaks']
                          [i:i+1], waves_peak['ECG_Q_Peaks'][i:i+1])
        qrs = qrs/sampling_rate
    except:
        print('Error: Not possible to calculate the qrs')
        pass
    
    return qrs*1000


def get_qrs_mean(signal, rpeaks, sampling_rate):
    """
    Get the qrs mean of the analyzed rr list
    ** Arguments
    * rpeaks: list of peaks
    * signal: dataset to analyze
    * sampling_rate = samplig rate of the signal
    """
    
    _, waves_peak = nk.ecg_delineate(signal, rpeaks, sampling_rate=sampling_rate)

    qrs = np.subtract(waves_peak['ECG_S_Peaks'], waves_peak['ECG_Q_Peaks'])

    qrs_mean = np.mean(qrs)
    qrs_is_nan = np.isnan(qrs_mean)

    if qrs_is_nan:
        qrs_mean = np.nanmean(qrs)

    qrs_mean = (qrs_mean/sampling_rate)*1000

    return qrs_mean
    


def get_pr(rpeaks, i, signal, sampling_rate):
    """
    Get the pr of the analyzed complex
    ** Arguments
    * rpeaks: list of peaks
    * i: iteration counter to get exact qtc of that data
    * signal: dataset to analyze
    * sampling_rate: sampling rate of the analyzed sample
    """

    signal_dwt, waves_dwt = nk.ecg_delineate(
        signal, rpeaks, sampling_rate, method="dwt", show=False, show_type='all')

    try:
        pr = waves_dwt['ECG_R_Onsets'][i] - waves_dwt['ECG_P_Offsets'][i]
        pr = pr/sampling_rate
    except:
        pr = 'Nan'

    return pr*1000


def pr_mean(rpeaks, signal, sampling_rate):
    """
    Get the pr mean of the dataset
    ** Arguments
    * rpeaks: list of peaks
    * signal: dataset to analyze
    * sampling_rate: sampling rate of the analyzed sample
    """

    signal_dwt, waves_dwt = nk.ecg_delineate(
    signal, rpeaks, sampling_rate, method="dwt", show=False, show_type='all')

    pr = np.subtract(waves_dwt['ECG_R_Onsets'],  waves_dwt['ECG_P_Offsets'])

    pr = np.divide(pr, sampling_rate)

    pr_mean = np.mean(pr)
    pr_is_nan = np.isnan(pr_mean)

    if pr_is_nan:
        pr_mean = np.nanmean(pr)

    return pr_mean*1000


def check_for_p_wave(signal, x, rpeaks):
    """
    It check for the p_waves, it require at least 2 qrs complex
    to analyze and exclude false p wave that are seen as t wave
    ** Arguments
    * signal: dataset to analyze
    * x: iteration counter to get exact qtc of that data
    * rpeaks: list of peaks 
    """
    sinus = True 

    # Analize every beat of the interval selected
    _, waves_peak = nk.ecg_delineate(signal, rpeaks)
    # Visualize the T-peaks, P-peaks, Q-peaks and S-peaks

    # Exclude the P peak if on same position of previous T Peak
    if x > 1:
        try:
            previous_t_peak = waves_peak['ECG_T_Peaks'][x-1:x][0]
            actual_p_peak = waves_peak['ECG_P_Peaks'][x:x+1][0]
            if actual_p_peak in range(previous_t_peak-20, previous_t_peak+20):
                waves_peak['ECG_P_Peaks'][x:x+1] = [float("nan")]
        except:
            pass

    p_wave = waves_peak['ECG_P_Peaks'][x:x+1]

    if np.isnan(p_wave):
        sinus = False

    return sinus