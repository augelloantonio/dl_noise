#importing basic libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tensorflow.keras.models import model_from_json


#importing datasets
test = pd.read_csv('mitbih_test.csv')
train = pd.read_csv('mitbih_train.csv')

test.fillna(0)
train.fillna(0)

#viewing normal dataset
test.head()

#dimenion for normal
test.shape

#dimension for abnormal
train.shape

#changing the random column names to sequential - normal
#as we have some numbers name as columns we need to change that to numbers as
for trains in train:
    train.columns = list(range(len(train.columns)))

#viewing edited columns for normal data
train.head()

#changing the random column names to sequential - abnormal
#as we have some numbers name as columns we need to change that to numbers as
for tests in test:
    test.columns = list(range(len(test.columns)))

#viewing edited columns for abnormal data
test.head()

#combining two data into one
#suffling the dataset and dropping the index
#As when concatenating we all have arranged 0 and 1 class in order manner
dataset = pd.concat([train, test], axis=0).sample(frac=1.0, random_state =0).reset_index(drop=True)
dataset.fillna(0)

#viewing combined dataset
dataset.head()

dataset.shape

#basic info of statistics
dataset.describe()

#basic information of dataset
dataset.info()

#viewing the uniqueness in dataset
dataset.nunique()

#skewness of the dataset
#the deviation of the distribution of the data from a normal distribution
#+ve mean > median > mode
#-ve mean < median < mode
dataset.skew()

#kurtosis of dataset
#identifies whether the tails of a given distribution contain extreme values
#Leptokurtic indicates a positive excess kurtosis
#mesokurtic distribution shows an excess kurtosis of zero or close to zero
#platykurtic distribution shows a negative excess kurtosis
dataset.kurtosis()
dataset = dataset.dropna()

#missing values any from the dataset
print(dataset.isnull().values.any())
print(str('Any missing data or NaN in the dataset:'), dataset.isnull().values.any())

nan_rows = dataset[dataset.isnull().T.any()]
print(nan_rows)


#data ranges in the dataset - sample
print("The minimum and maximum values are {}, {}".format(np.min(dataset.iloc[-2,:].values), np.max(dataset.iloc[-2,:].values)))

#correlation for all features in the dataset
correlation_data =dataset.corr()
print(correlation_data)

import seaborn as sns
#visulaization for correlation
# plt.figure(figsize=(10,7.5))
# sns.heatmap(correlation_data, annot=True, cmap='BrBG')

#for target value count
label_dataset = dataset[187].value_counts()
label_dataset

#visualization for target label
label_dataset.plot.bar()

#splitting dataset to dependent and independent variable
X = dataset.iloc[:,:-1].values #independent values / features
y = dataset.iloc[:,-1].values #dependent values / target

#checking imbalance of the labels
from collections import Counter
counter_before = Counter(y)
print(counter_before)

#applying SMOTE for imbalance
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

#after applying SMOTE for imbalance condition
counter_after = Counter(y)
print(counter_after)

#splitting the datasets for training and testing process
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3, random_state=42)

#size for the sets
print('size of X_train:', X_train.shape)
print('size of X_test:', X_test.shape)
print('size of y_train:', y_train.shape)
print('size of y_test:', y_test.shape)

#CNN
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout
#Reshape train and test data to (n_samples, 187, 1), where each sample is of size (187, 1)
X_train = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)

print("X Train shape: ", X_train.shape)
print("X Test shape: ", X_test.shape)

# Create sequential model 
cnn_model = tf.keras.models.Sequential()
#First CNN layer  with 32 filters, conv window 3, relu activation and same padding
cnn_model.add(Conv1D(filters=32, kernel_size=(3,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001), input_shape = (X_train.shape[1],1)))
#Second CNN layer  with 64 filters, conv window 3, relu activation and same padding
cnn_model.add(Conv1D(filters=64, kernel_size=(3,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
#Third CNN layer with 128 filters, conv window 3, relu activation and same padding
cnn_model.add(Conv1D(filters=128, kernel_size=(3,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
#Fourth CNN layer with Max pooling
cnn_model.add(MaxPool1D(pool_size=(3,), strides=2, padding='same'))
cnn_model.add(Dropout(0.5))
#Flatten the output
cnn_model.add(Flatten())
#Add a dense layer with 256 neurons
cnn_model.add(Dense(units = 256, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
#Add a dense layer with 512 neurons
cnn_model.add(Dense(units = 512, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
#Softmax as last layer with five outputs
cnn_model.add(Dense(units = 5, activation='softmax'))

cnn_model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

cnn_model.summary()

cnn_model_history = cnn_model.fit(X_train, y_train, epochs=10, batch_size = 10, validation_data = (X_test, y_test))


# evaluate the model
print("********")
scores = cnn_model.evaluate(X_train, y_train, verbose=0)
print("%s: %.2f%%" % (cnn_model.metrics_names[1], scores[1]*100))
print("********")

# serialize model to YAML
model_yaml = cnn_model.to_json()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
cnn_model.save_weights("model.h5")
print("Saved model to disk")

# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_json(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_train, y_train, verbose=1)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

""" 
plt.plot(cnn_model_history.history['accuracy'])
plt.plot(cnn_model_history.history['val_accuracy'])
plt.legend(["accuracy","val_accuracy"])
plt.title('Accuracy Vs Val_Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()

plt.plot(cnn_model_history.history['loss'])
plt.plot(cnn_model_history.history['val_loss'])
plt.legend(["loss","val_loss"])
plt.title('Loss Vs Val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.show()
"""