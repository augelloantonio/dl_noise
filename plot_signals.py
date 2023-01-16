import matplotlib.pyplot as plt
import pandas as pd
import time

""" test = pd.read_csv('good_signal_for_ml.csv')
df = test.reset_index()  # make sure indexes pair with number of rows

for index, row in df.iterrows():
    if index >= 200:
        print(index+1)
        plt.xticks([])
        plt.plot(row[1:])
        plt.show()
        time.sleep(1.5) """

test = pd.read_csv('artifact_signal_for_ml.csv')
df = test.reset_index()  # make sure indexes pair with number of rows

for index, row in df.iterrows():
    if index >= 86:
        print(index+1)
        plt.xticks([])
        plt.plot(row[1:])
        plt.show()
        time.sleep(1.5)