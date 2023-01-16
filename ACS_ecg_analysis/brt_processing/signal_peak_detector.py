import numpy as np
import math

def peakDetector(y, samplingRate, lag, threshold, influence):

    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter[i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = 0

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals=np.asarray(signals),
                avgFilter=np.asarray(avgFilter),
                stdFilter=np.asarray(stdFilter))


def findPeakSignal(signal):
    trace = []
    for i in signal:
        if i>0:
        #if i>-100:
        #if i >-5000:
        #if i>-0.1:
            trace.append(300)
        else:
            trace.append(0)

    """ 
    # print(len(trace))
    prev_max = 0
    actual_max = 0
    i=0
    count=0
    while i<len(trace)-2:
        if trace[i]>0 and trace[i+1]>0:
            count+=1
        else:
            try:
                if count>0 and count<30:
                    # print("interval: " + str(i-count) + ":" + str(i+count))
                    for x in range(i-count, i+count):
                        trace[x]=0
            except:
                pass
            
            count=0
        i+=1
        
    i=0
    count=0
    while i<len(trace)-2:

        if trace[i]==0 and trace[i+1]==0:
            count+=1
        else:
            if count>0 and count<40:
                for x in range(i-count, i+1):
                    if np.mean(signal[i-count:i+count])>0:
                        trace[x]=2000
            count=0
        
        if trace[i]>0 and trace[i+1]>0:
            count+=1
        else:
            if (len(signal[i-count:i]))>0:
                actual_max = np.amax(signal[i-count:i])
                if actual_max<0.25*prev_max:
                    for x in range(i-count, i+1):
                        trace[x]=0
                else:
                    prev_max=actual_max
            count=0

        i += 1  """
    return trace


def countPeaks(peakSignal):
    i=0
    count=0
    while i < len(peakSignal)-1:
        if peakSignal[i] >0 and peakSignal[i+1] == 0:
            count+=1
        i+=1

    return count