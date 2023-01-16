from ACS_ecg_analysis.ecg_processing import *
from hrvanalysis import *

def get_hrv_frequency_measures(nni):
    """
    Get Frequency HRV Measures using HRV-Analysis

    **ARGS
    * nni: list of beat to beat interval
    """

    # HRV ANALYSIS on BIOSPPY and PyHRV
    frequency_measures = get_frequency_domain_features(nni)

    return frequency_measures