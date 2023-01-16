import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import datetime

import ACS_ecg_analysis
from ACS_ecg_analysis.run import run_stress_analysis, run_new_ecg_analysis

path = "high_artifacts.ecg"

ecg, signal_average_mean_removed, finalEcgArtifactRemoved, parameter, anomalies_time, time_domain, anomalies_indexing, peaks, thrheshold_list, corr_peaks, beatClassificationList = run_new_ecg_analysis(path)

# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_json(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# Check its architecture
loaded_model.summary()

CATEGORIES = ["NORMAL", "NOISE"]

# plt.plot(signal_average_mean_removed, label="Mean Removed")

filtSignal=[]
for i in signal_average_mean_removed:
    filtSignal.append(i)

for i in range(1, len(signal_average_mean_removed)):
    if i%1250==0:
        ecg_to_analyze = signal_average_mean_removed[i-1250:i]
        ecg_to_analyze=np.expand_dims(ecg_to_analyze, axis=0) # will move it to (1,28,28)
        prediction = loaded_model.predict(ecg_to_analyze)
        # print("Predicted the classes: " , prediction)
        # print(ecg_to_analyze)
        print("PREDICTED")
        print(np.argmax(prediction))
        if np.argmax(prediction) == 1:
            # plt.plot(prediction[0])
            for x in range(i-1250, i):
                filtSignal[x] = 0

plt.plot(filtSignal, label="IA REMOVED")
# plt.plot(finalEcgArtifactRemoved, label="OLD METHOD")
plt.legend()
plt.show()

# plt.plot(filtSignal, label="IA REMOVED")
plt.plot(finalEcgArtifactRemoved, label="OLD METHOD")
plt.legend()
plt.show()

i = 0
count = 0
for i in filtSignal:
    if i == 0:
        count += 1
        percentageRemoved = ((len(ecg)-count)/len(ecg))*100
        if percentageRemoved > 35:
            durationEcg = str(datetime.timedelta(seconds=len(ecg)/250))
            durationArtifacts = str(datetime.timedelta(seconds=count/250))
print("Duration ia artifact removed: " + str(durationArtifacts))

i = 0
count = 0
for i in finalEcgArtifactRemoved:
    if i == 0:
        count += 1
        percentageRemoved = ((len(ecg)-count)/len(ecg))*100
        if percentageRemoved > 35:
            durationEcg = str(datetime.timedelta(seconds=len(ecg)/250))
            durationArtifacts = str(datetime.timedelta(seconds=count/250))
print("Duration our artifact removed: " + str(durationArtifacts))