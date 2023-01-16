import wfdb

def loadAnnotationSample(filename):
    #ANNOTATIONS

    # file = filename[:12]
    file = filename[9:-8]
    
    #print(file)

    annotation = wfdb.rdann('/Users/antonioaugello/Desktop/projects/ecg_analisys/data/mit-bih-arrhythmia-database-1.0.0/' + file, 'atr')
    
    # annotation = wfdb.rdann('/Users/antonioaugello/Desktop/projects/ecg_analisys/mit_regular/' + file, 'atr')

    ann = annotation.sample

    return ann