import numpy as np
import math


def meanRemoval(data):

    newData = []

    for i in data:
        newData.append(i)

    meanBrt = np.median(newData)
    i = 0

    while i < len(newData):
        newData[i] = newData[i]-meanBrt

        i += 1

    return newData


def bandpassFilt(sample, n, samplingRate, freq_1, freq_2):
    sampletoFilt = []

    for i in sample:
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


def movingAverageMeanBrt(data):
    newData = []
    for i in data:
        newData.append(i)

    # 25 for full sample
    size = 250
    i = 0
    moving_averages = []

    while i < len(newData)-size+1:
        if not math.isnan(i):

            this_window = data[i: i+size]
            window_average = np.sum(this_window) / size

            val = newData[i]-window_average
            moving_averages.append(val)

        i += 1

    # moving_averages = np.nan_to_num(moving_averages, nan=0.0)

    return moving_averages


def Butterworth_Lowpass(sample, n, samplingRate, freq):

    filtSample = []

    for i in sample:
        if not math.isnan(i):
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


def filterVisualizeBRT(brt, fs):

    if len(brt) > fs/2:
        lowPassRemoved = Butterworth_Lowpass(brt, 4, fs/2, 4)
        meanRemoved = movingAverageMeanBrt(lowPassRemoved)

        return meanRemoved
    else:
        return "The signal lenght is too short to be filtered."
