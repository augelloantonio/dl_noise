from ACS_ecg_analysis.brt_processing.signal_peak_detector import *
from ACS_ecg_analysis.brt_processing.signal_filtering import *


def main_brt(fileName):

    brt = np.genfromtxt(fileName, delimiter=',')

    """ lowPassRemoved = Butterworth_Lowpass(brt, 4, 128, 4)
    meanRemoved = movingAverageMeanBrt(lowPassRemoved) """

    newbrtpeaks = findPeakSignal(brt)

    meanBreath = countPeaks(newbrtpeaks)/((len(brt)/250/60))

    print(meanBreath)
    return brt, brt, newbrtpeaks, meanBreath


def calculateBrtFromRawData(brt, fs):

    lowPassRemoved = Butterworth_Lowpass(brt, 4, 128, 4)
    meanRemoved = movingAverageMeanBrt(lowPassRemoved)

    newbrtpeaks = findPeakSignal(meanRemoved)
    meanBreath = countPeaks(newbrtpeaks)/((len(brt)/250/60))
    print(int(meanBreath))
        
    return brt, meanRemoved, newbrtpeaks, meanBreath


def calculateBrtFromFitltereSignal(fileName, fs):

    brt = np.genfromtxt(fileName, delimiter=',')
    newbrtpeaks = findPeakSignal(brt)
    print(newbrtpeaks)
    meanBreath = countPeaks(newbrtpeaks)/((len(brt)/250/60))
        
    return brt, newbrtpeaks, meanBreath