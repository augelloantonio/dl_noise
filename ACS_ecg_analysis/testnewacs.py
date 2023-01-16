def newACSPeakDetector(signal, fs):
    start_time = time.time()
    
    signalPeaks = [0]
    missedPeaks = [0]

    SPKI = 0.0
    NPKI = 0.0

    thresholdI1 = 0.0
    thresholdI2 = 0.0

    RRMissed = 0
    RRMissedPrev = 0
    index = 0

    peaks = []
    thrheshold_list = []
    SPKI_list = []
    NPKI_list = []
    
    #LEARNING PHASE  
    #2 second initialize phase for MIW, 25% of max amplitude considered signal, 50% of mean signal considered noise
    initializeTime = 2 * fs 
    SPKI = max(signal[:initializeTime]) * 0.25 
    NPKI = np.mean(signal[:initializeTime]) * 0.5 
    THRESHOLDI1 = NPKI + 0.25 * (SPKI-NPKI)
    THRESHOLDI2 = 0.5 * THRESHOLDI1 

    for i in range(len(signal)):
        if i > 0 and i < len(signal)-1:
            if signal[i-1] < signal[i] and signal[i+1] < signal[i]:
                peaks.append(i)

    i = 1
    while i < len(peaks)-2:
        
        if len(signalPeaks)>2 and RRMissed>0:
            if peaks[i]- signalPeaks[-1]>RRMissed:
            
                RRMissedPrev = RRMissed

                thresholdI2 = 0.15*thresholdI1
                x = signalPeaks[-1]
                y = peaks[i]
                actual_signal = signal[x:y]
                new_peaks = []
                                
                for p in range(len(actual_signal)):
                    if len(actual_signal)>0:
                        if p > 0 and p < len(actual_signal)-1:
                            if actual_signal[p-1] < actual_signal[p] and actual_signal[p+1] < actual_signal[p]:
                                new_peaks.append(p)
                        
                z=1
                while z < len(new_peaks)-2:
                    if actual_signal[new_peaks[z]] > thresholdI2 and actual_signal[new_peaks[z]] > actual_signal[new_peaks[z-1]] and actual_signal[new_peaks[z]] > actual_signal[new_peaks[z+1]]:
                        if new_peaks[z] - missedPeaks[-1] > 0.36*fs:
                            missedPeaks.append(new_peaks[z]+signalPeaks[-1])
                            SPKI = 0.125*actual_signal[new_peaks[z]] + 0.875*SPKI
                            print(new_peaks[z])
                        if new_peaks[z] - missedPeaks[-1] >0.2*fs and new_peaks[z] - missedPeaks[-1] < 0.36*fs and actual_signal[new_peaks[z]] > 0.5*actual_signal[missedPeaks[-1]]:
                            missedPeaks.append(new_peaks[z]+signalPeaks[-1])
                            SPKI = 0.125*actual_signal[new_peaks[z]] + 0.875*SPKI
                            print(new_peaks[z])
                    else:
                        NPKI = 0.125*actual_signal[new_peaks[z]] + 0.875*NPKI
                        
                    thresholdI2 = NPKI + 0.25*(SPKI - NPKI)
                    z+=1
            RRMissed = 0
            
            
            
        if signal[peaks[i]] > thresholdI1 and signal[peaks[i]] > signal[peaks[i-1]] and signal[peaks[i]] > signal[peaks[i+1]]:
            if peaks[i] - signalPeaks[-1] > 0.36*fs:
                signalPeaks.append(peaks[i])
                SPKI = 0.125*signal[peaks[i]] + 0.875*SPKI
            if peaks[i] - signalPeaks[-1] >0.2*fs and peaks[i] - signalPeaks[-1] < 0.36*fs and signal[peaks[i]] > 0.5*signal[signalPeaks[-1]]:
                signalPeaks.append(peaks[i])
                SPKI = 0.125*signal[peaks[i]] + 0.875*SPKI

        else:
            NPKI = 0.125*signal[peaks[i]] + 0.875*NPKI

        SPKI_list.append(SPKI)
        NPKI_list.append(NPKI)

        thresholdI1 = NPKI + 0.25*(SPKI - NPKI)
        thrheshold_list.append(thresholdI1)

        # To fix in case there is a blank area in the between the array
        if len(signalPeaks)>8:
            array=signalPeaks[len(
                            signalPeaks) - 9: len(signalPeaks) - 2]
            
            RR=diffInt(array, fs)

            RRAve=sum(RR)/len(RR)
            RRMissed = (1.66 * RRAve)
            
            x = 1
            while x<len(array):
                if array[x]-array[x-1]>2500:
                    RRMissed = RRMissedPrev
                x+=1
        i += 1
    i = 1
    
    # Noticed to fail peak detection on Accyourate signals - but ok on mit db
    while i < len(signalPeaks)-1:
        if signalPeaks[i]-signalPeaks[i-1] < fs/1000*fs:
            if signal[signalPeaks[i]]>signal[signalPeaks[i-1]]:
                signalPeaks.pop(i-1)
            else:
                signalPeaks.pop(i) 
        i+=1
            
    print(missedPeaks)
    for i in missedPeaks:
        signalPeaks.append(i)
    
    signalPeaks.sort()
    
    #print((time.time() - start_time))
        
    return signalPeaks[1:], thrheshold_list