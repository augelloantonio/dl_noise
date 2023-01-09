#importing basic libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#importing datasets
mitbih_test = pd.read_csv('mitbih_test.csv', header=None)
mitbih_test = mitbih_test.dropna()

mitbih_train = pd.read_csv('mitbih_train.csv', header=None)
mitbih_train = mitbih_train.dropna()

mitbih_train_0=mitbih_train[mitbih_train.iloc[: , -1]==0.0]

mitbih_train_1=mitbih_train[mitbih_train.iloc[: , -1]==1.0]

from sklearn.utils import resample

value_counts=mitbih_train.iloc[:,mitbih_train.shape[1]-1].value_counts()

mitbih_train_0=mitbih_train[mitbih_train.iloc[:,mitbih_train.shape[1]-1]==0]
mitbih_train_1=mitbih_train[mitbih_train.iloc[:,mitbih_train.shape[1]-1]==1]
mitbih_train_2=mitbih_train[mitbih_train.iloc[:,mitbih_train.shape[1]-1]==2]
mitbih_train_3=mitbih_train[mitbih_train.iloc[:,mitbih_train.shape[1]-1]==3]
mitbih_train_4=mitbih_train[mitbih_train.iloc[:,mitbih_train.shape[1]-1]==4]

mitbih_train_1_upsample=resample(mitbih_train_1,replace=True,n_samples=mitbih_train_0.shape[0],random_state=123)

mitbih_train_upsampled=np.concatenate((mitbih_train_0,mitbih_train_1_upsample))
mitbih_train_upsampled.shape

mitbih_train_upsampled=pd.DataFrame(mitbih_train_upsampled)
mitbih_train_upsampled.iloc[:,mitbih_train_upsampled.shape[1]-1].value_counts()

y=mitbih_train_upsampled.iloc[:,mitbih_train_upsampled.shape[1]-1]
X=mitbih_train_upsampled.iloc[:,0:(mitbih_train_upsampled.shape[1]-1)]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score as acc


knn = KNN()
from sklearn.model_selection import GridSearchCV
k_range = list(range(1, 15))
param_grid = dict(n_neighbors=k_range)
  
# defining parameter range
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)
  
# fitting the model for grid search
grid_search=grid.fit(x_train, y_train)

print(grid_search.best_params_)
accuracy = grid_search.best_score_ *100
print(accuracy)

# help(knn.fit)

knn = KNN(n_neighbors=2)
knn.fit(x_train, y_train)

preds=knn.predict(x_test)

print('Accuracy KNN: %f' % (acc(y_test, preds)))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, preds))