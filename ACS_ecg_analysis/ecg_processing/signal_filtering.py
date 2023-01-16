import numpy as np
import math
from scipy import signal

def filterVisualizeEcg(data):
    """
    Normalize the raw ECG signal that gets ready for the plotting.

    * ARGS:
    - data: raw ecg as list

    * Returns:
    - finalEcgArtifactRemoved: filtered ecg as list
    """
    fs = 250

    signal_average_mean_removed = movingAverageMean(data, fs)
    finalEcgArtifactRemoved = artifactRemoval(signal_average_mean_removed)
    
    return finalEcgArtifactRemoved

def checkSignalValidity(data):
    """
    Checks if YouCare ECG signal is valid.
    The validity is checked every 5 seconds of ECG (1250 samples).
    The sample mean is detected, if not in range -20, 20 then is not valid.

    * ARGS:
    - data: raw ecg as list

    * Returns:
    - isValid: boolean
    """
    isValid = True

    i = 1
    while i < len(data):
        if i%1250==0:
            meanA = (np.median(data[i-1250:i]))
            if (round(meanA) not in range(-20,20)):
                isValid = False
            else:
                isValid = True
        i+=1

    return isValid


def valuesOfIndexes(array, indexes):
    values = []
    for i in indexes:
        values.append(array[i])

    return values

def diffInt(arr):
    diff = []

    i = 0
    while i < len(arr)-1:
        diff.append(arr[i + 1] - arr[i])
        i += 1

    return diff


def movingAverageMean(data, size):
    """
    Moving average mean filter.

    * ARGS:
    - data: raw ecg as list
    - size: the size of the window to run

    * Returns:
    - moving_averages: the moving average signal as list
    """
    newData = []
    for i in data:
        newData.append(i)

    i = 0
    moving_averages = []

    while i < len(newData)-size+1:
        this_window = data[i: i+size]
        window_average = np.sum(this_window) / size

        val = newData[i]-window_average
        moving_averages.append(val)

        i += 1

    # moving_averages = np.nan_to_num(moving_averages, nan=0.0)

    return moving_averages


def movingAverageMeanPamTompkins(data, fs):
    """
    Moving average mean filter as Pam Tompkins standard.

    * ARGS:
    - data: raw ecg as list
    - size: the size of the window to run

    * Returns:
    - window_average_list: the moving average signal as list
    """
    newData = []
    for i in data:
        newData.append(i)

    size = int(0.15*fs)  # PREV 150
    i = 0
    moving_averages = []
    window_average_list = []

    while i < len(newData)-size+1:
        this_window = data[i: i+size]
        window_average = np.sum(this_window) / size
        window_average_list.append(window_average)

        i += 1

    return window_average_list


def meanRemoval(data):
    """
    Removes the mean from a signal.

    * ARGS:
    - data: raw signal as list

    * Returns:
    - newData: the signal with mean removed as list
    """
    newData = []

    for i in data:
        newData.append(i)

    meanBrt = np.median(newData)
    i = 0

    while i < len(newData):
        newData[i] = newData[i]-meanBrt

        i += 1

    return newData


def lowPass(data, f2):
    """
    Smple low pass filter.

    * ARGS:
    - data: raw signal as list
    - f2: cut off frequency

    * Returns:
    - newData: the signal with low pass filtered
    """
    newData = []

    for i in data:
        newData.append(i)

    imf = []

    mean = np.mean(newData)

    for i in range(0, len(newData)):
        imf.append(newData[i] - mean)

    for i in range(1, len(newData)):
        newData[i] = newData[i] - imf[i]*f2

    return newData


def Butterworth_Lowpass(sample, n, samplingRate, freq):
    """
    Butterworth Low Pass filter.

    * ARGS:
    - sample: raw signal as list
    - n number of iterations
    - samplingRate: sampling frequency
    - freq: cut off frequency

    * Returns:
    - newData: the signal with low pass filtered
    """
    filtSample = []

    for i in sample:
        filtSample.append(i)

    n = n / 2

    a = math.sin(math.pi * freq / samplingRate)
    a2 = a * a

    A = []
    d1 = []
    d2 = []
    w0 = []
    w1 = []
    w2 = []

    for i in range(0, int(n)):
        A.append(0.0)
        d1.append(0.0)
        d2.append(0.0)
        w0.append(0.0)
        w1.append(0.0)
        w2.append(0.0)

    for i in range(0, int(n)):
        r = math.sin(math.pi * (2.0 * (i) + 1.0) / (4.0 * (n)))
        s = a2 + 2.0*a*r + 1.0
        A[i] = (a2/s)
        d1[i] = (2.0*(1-a2)/s)
        d2[i] = (-(a2 - 2.0*a*r + 1.0)/s)

    for x in range(0, len(filtSample)-1):
        for i in range(0, int(n)):
            w0[i] = d1[i]*w1[i] + d2[i]*w2[i] + filtSample[x]

            filtSample[x] = (A[i] * (w0[i] + 2.0*w1[i] + w2[i]))

            w2[i] = w1[i]
            w1[i] = w0[i]

    return filtSample


def Butterworth_Highpass(sample, n, samplingRate, freq):
    """
    Butterworth High Passs filter.

    * ARGS:
    - sample: raw signal as list
    - n number of iterations
    - samplingRate: sampling frequency
    - freq: cut off frequency

    * Returns:
    - filtSample: the signal with low pass filtered
    """
    filtSample = []

    for i in sample:
        filtSample.append(i)

    n = n / 2

    a = math.tan(math.pi * freq / samplingRate)
    a2 = a * a

    A = []
    d1 = []
    d2 = []
    w0 = []
    w1 = []
    w2 = []

    for i in range(0, int(n)):
        A.append(0.0)
        d1.append(0.0)
        d2.append(0.0)
        w0.append(0.0)
        w1.append(0.0)
        w2.append(0.0)

    for i in range(0, int(n)):
        r = math.sin(math.pi * (2.0 * (i) + 1.0) / (4.0 * (n)))
        s = a2 + 2.0*a*r + 1.0
        A[i] = (1.0 / s)
        d1[i] = (2.0 * (1 - a2) / s)
        d2[i] = (-(a2 - 2.0*a*r + 1.0) / s)

    for x in range(0, len(filtSample)-1):
        for i in range(0, int(n)):
            w0[i] = d1[i]*w1[i] + d2[i]*w2[i] + filtSample[x]

            filtSample[x] = (A[i] * (w0[i] - 2.0*w1[i] + w2[i]))

            w2[i] = w1[i]
            w1[i] = w0[i]

    return filtSample


def bandpassFilt(sample, n, samplingRate, freq_1, freq_2):
    """
    Bandpass filter, it does a High Pass & Low Pass filtering.

    * ARGS:
    - sample: raw signal as list
    - n number of iterations
    - samplingRate: sampling frequency
    - freq_1: high pass cut off frequency
    - freq_2: low pass cut off frequency

    * Returns:
    - sampletoFilt: the signal with low pass filtered
    """
    sampletoFilt = []

    for i in sample: 
        if not math.isnan(i):
            sampletoFilt.append(i)

    n = int(n / 4)

    a = math.cos(math.pi*(freq_1+freq_2)/samplingRate) / \
        math.cos(math.pi*(freq_1-freq_2)/samplingRate)
    a2 = a * a

    b = math.tan(math.pi * (freq_1 - freq_2) / samplingRate)
    b2 = b * b

    A = []
    d1 = []
    d2 = []
    d3 = []
    d4 = []

    for i in range(0, n+1):
        A.append(0.0)
        d1.append(0.0)
        d2.append(0.0)
        d3.append(0.0)
        d4.append(0.0)

    for i in range(0, n+1):

        r = math.sin(math.pi * (2.0*(i) + 1.0) / (4.0 * (n)))
        s = b2 + 2.0*b*r + 1.0

        A[i] = (b2 / s)

        d1[i] = (4.0 * a * (1.0 + b*r) / s)

        d2[i] = (2.0 * (b2 - 2.0*a2 - 1.0) / s)
        d3[i] = 4.0 * a * (1.0 - b*r) / s
        d4[i] = (-(b2 - 2.0*b*r + 1.0) / s)

    w0 = []
    w1 = []
    w2 = []
    w3 = []
    w4 = []

    for i in range(0, n+1):
        w0.append(0.0)
        w1.append(0.0)
        w2.append(0.0)
        w3.append(0.0)
        w4.append(0.0)

    for x in range(0, len(sampletoFilt)-1):
        for i in range(0, n+1):

            w0[i] = (d1[i]*w1[i] + d2[i]*w2[i] + d3[i] *
                     w3[i] + d4[i]*w4[i] + sampletoFilt[x])

            sampletoFilt[x] = A[i] * (w0[i] - 2.0*w2[i] + w4[i])

            w4[i] = w3[i]
            w3[i] = w2[i]
            w2[i] = w1[i]
            w1[i] = w0[i]

    return sampletoFilt


def artifactRemoval(signal, fs):
    """
    ECG artifact removal filter.

    * ARGS:
    - signal: raw signal as list
    - fs: sampling frequency

    * Returns:
    - filtSignal: the signal with low pass filtered
    """
    filtSignal = []

    for i in signal:
        filtSignal.append(i)

    i = 1
    while i <= len(signal):
        if i % (fs*5) == 0:
            max_signal = np.max(signal[i-(fs*5):i])
            std = np.std(signal[i-(fs*5):i])
            # print("MAX " + str(max_signal))
            # print("STD " + str(std)) 
            if std > 1:
                for x in range(i-(fs*5), i):
                    filtSignal[x] = 0
            if max_signal < 0.6 or max_signal > 3.0:
                for x in range(i-(fs*5), i):
                    filtSignal[x] = 0
       
        i += 1

    max_signal = np.max(signal[len(signal)-(fs*5):len(signal)])
    std = np.std(signal[len(signal)-(fs*5):len(signal)])
    if std > 0.5:
        for x in range(len(signal)-(fs*5), len(signal)):
            filtSignal[x] = 0
    if max_signal < 0.6 or max_signal > 6.0:
        for x in range(len(signal)-(fs*5), len(signal)):
            filtSignal[x] = 0
    return filtSignal

def sliceForIndex(signal):
    data = []
    resultData = []
    resultdata = []

    for i in signal:
        data.append(i)

    i = 1
    while i <= len(signal):
        if i % 2500 == 0:
            movinaveragedata = movingAverageMean(data[i-2500:i], 250)
            resultdata = artifactRemoval(movinaveragedata)

            for x in resultdata:
                resultData.append(x)
        i += 1

    return resultData


def derivateStep(data,fs):
    derivate = []

    for i in range(1, len(data)-2):
        derivate.append(
            (1/8*fs) * ((-data[i-2]-2*data[i-1])+2*data[i+1]+data[i+2]))

    return derivate


def newfilter(ECG, fs):
    if type(ECG) == list or type(ECG) is np.ndarray:
        ECG = np.array(ECG)             
        
    #Initialize
    RRAVERAGE1 = []
    RRAVERAGE2 = []
    IWF_signal_peaks = []
    IWF_noise_peaks = []
    noise_peaks = []
    ECG_bp_peaks = np.array([])
    ECG_bp_signal_peaks = []
    ECG_bp_noise_peaks = []
    final_R_locs = []
    T_wave_found = 0      
    
    #LOW PASS FILTERING
    #Transfer function: H(z)=(1-z^-6)^2/(1-z^-1)^2
    a = np.array([1, -2, 1])
    b = np.array([1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1])   
        
    impulse = np.repeat(0., len(b)); impulse[0] = 1.    
    impulse_response = signal.lfilter(b,a,impulse)
    
    #convolve ECG signal with impulse response
    ECG_lp = np.convolve(impulse_response, ECG)
    ECG_lp = ECG_lp / (max(abs(ECG_lp)))
    delay = 12 #full convolution
    
    #HIGH PASS FILTERING
    #Transfer function: H(z)=(-1+32z^-16+z^-32)/(1+z^-1)
    a = np.array([1, -1])           
    b = np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 32, -32, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, -1])
                  
    impulse = np.repeat(0., len(b)); impulse[0] = 1.    
    impulse_response = signal.lfilter(b,a,impulse)
    
    ECG_lp_hp = np.convolve(impulse_response, ECG_lp)
    ECG_lp_hp = ECG_lp_hp/(max(abs(ECG_lp_hp)))
    delay = delay + 32 
    
    #BAND PASS FILTER 
    nyq = fs / 2        
    lowCut = 5 / nyq  #cut off frequencies are normalized from 0 to 1, where 1 is the Nyquist frequency
    highCut = 15 / nyq
    order = 5
    b,a = signal.butter(order, [lowCut, highCut], btype = 'bandpass')
    ECG_bp = signal.lfilter(b, a, ECG_lp_hp)
    
    #DIFFERENTIATION
    #Transfer function: H(z)=(1/8T)(-z^-2-2z^-1+2z^1+z^2)
    T = 1/fs
    b = np.array([-1, -2, 0, 2, 1]) * (1 / (8 * T))
    a = 1
    #Note impulse response of the filter with a = [1] is b
    ECG_deriv = np.convolve(ECG_bp, b)
    delay = delay + 4 
    
    #SQUARING FUNCTION
    ECG_squared = ECG_deriv ** 2
    
    #MOVING INTEGRATION WAVEFORM 
    N = int(np.ceil(0.150 * fs)) 
    ECG_movavg = np.convolve(ECG_squared,(1 / N) * np.ones((1, N))[0])
        
    return ECG_movavg