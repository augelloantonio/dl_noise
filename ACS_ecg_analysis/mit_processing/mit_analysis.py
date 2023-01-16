#################
#
# Check Positive
#
#################
def checkPositive(ann, peaks):
    realPeaks = []
    fakePositive = []
    i = 0
    lenght = 0
    if len(peaks) > len(ann):
        lenght = len(peaks)
        arr = peaks
        arr2 = ann
    else:
        lenght = len(ann)
        arr = ann
        arr2 = peaks

    for i in peaks:
        for n in ann:
            if i+75 > n > i-75:
                realPeaks.append(i)

    for i in peaks:
        if i not in realPeaks:
            fakePositive.append(i)

    #print("Fake +")
    print(len(fakePositive))
    # print(fakePositive)
    return fakePositive

#################
#
# Check Negative
#
#################
# start from annotation and subtract number of real annotation from the presents
def checkNegative(ann, peaks):
    fakePeaks = []
    fakeNegative = []

    # [[L5[l2 - 1] * sl1 for sl1, l3 in zip(l1, L3) for l2 in L2 if L4[l2 - 1] == l3] for l1 in L1]
 
    for i in ann:
        for n in peaks:
            if i+80 > n > i-80: #try with 95
                fakePeaks.append(i)

    for i in ann:
        if i not in fakePeaks:
            fakeNegative.append(i)

    # print("fake -")
    print(len(fakeNegative))    
    return fakeNegative