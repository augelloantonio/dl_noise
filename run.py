import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt

ecg = np.genfromtxt("1611652801.ecg", delimiter="")
print(len(ecg))

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

plt.plot(ecg)
plt.show()

for i in range(1, len(ecg)):
    if i%1250==0:
        ecg_to_analyze = ecg[i-1250:i]
        ecg_to_analyze=np.expand_dims(ecg_to_analyze, axis=0) # will move it to (1,28,28)
        prediction = loaded_model.predict(ecg_to_analyze)
        # print("Predicted the classes: " , prediction)
        # print(ecg_to_analyze)
        print("PREDICTED")
        print(np.argmax(prediction))
        if np.argmax(prediction) ==1:
            # plt.plot(prediction[0])
            plt.plot(ecg_to_analyze[0])
            plt.show()

