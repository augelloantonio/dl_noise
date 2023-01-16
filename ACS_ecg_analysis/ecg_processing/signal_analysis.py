import numpy as np
import math

def calculateNNI(rpeaks):
    """
    NNI calculation starting from the found r peaks

    * ARGS:
    - rpeaks: list of found peaks

    * Returns:
    - nni: list of beat to beat interval
    """
    nniFiltered = []
    try:
        nni = []
        nni = np.diff(rpeaks)
        nni = np.divide(nni, 250)
        nni = np.multiply(nni, 1000)
        i = 0

        for i in nni:
            if int(i) > 200 and int(i) < 1800:
                nniFiltered.append(i)
    except:
        pass

    return nniFiltered


def calculateBpm(nni):
    """
    Calculate the Heart Rate from the NNI list

    * ARGS:
    - nni: list of beat to beat interval

    * Returns:
    - bpm: heart rate mean
    """

    bpm = 0
    r = []
    hrv = []
    if len(nni) > 2:
        for i in range(0, len(nni)):
            r.append(60000 / nni[i])
    
    i = 1
    filtered_bpm = []

    while i<len(r)-1:
        if r[i-1]<1.5*r[i] and r[i+1]<1.5*r[i]:
            r.remove(r[i])
        i+=1

    bpm = np.mean(r)

    return bpm, r


def calculateRmssd(nni):
    """
    Calculate the incorrect RMSSD to calculate ASF

    * ARGS:
    - nni: list of beat to beat interval

    * Returns:
    - rmssd
    """
    rmssd = 0
    if len(nni) > 2:
        nni = np.array([i for i in nni if i in range(200, 1800)])
        diff_nni_rmssd = np.diff(nni)
        diff_nni_pow_rmssd = np.power(diff_nni_rmssd, 2)
        diff_nni = np.divide(diff_nni_pow_rmssd, (nni.size - 2))
        diff_nni_somm_rmssd = np.mean(diff_nni)
        rmssd = np.sqrt(diff_nni_somm_rmssd)

    return rmssd


def calculateRmssd2(nni):
    """
    Calculate the correct RMSSD calculation.


    * ARGS:
    - nni: list of beat to beat interval

    * Returns:
    - rmssd
    """

    if len(nni)>2:
        diff_nni_rmssd = np.diff(nni)
        diff_nni_pow_rmssd = np.power(diff_nni_rmssd, 2)
        sum_nni = np.sum(diff_nni_pow_rmssd)
        diff_nni = np.divide(sum_nni, (len(nni) - 1))
        #diff_nni_somm_rmssd = np.mean(diff_nni)

        rmssd = np.sqrt(diff_nni)

        if not np.isnan(rmssd):
            rmssd = 0
    else:
        rmssd = 0

    return rmssd
