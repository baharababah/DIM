# -*- coding: utf-8 -*-
"""Save_LSTM_Model_Light.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uu6RgATMbbcDPeDKA4bIpqDGPI-rW1D_
"""

import numpy as np
import pandas as pd
import io
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from google.colab import files
from keras.models import Sequential
from keras.layers import Dense, ReLU, LSTM, GRU
import tensorflow as tf                # To remove logging
from statistics import mode
from keras.models import model_from_json

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
pd.options.mode.chained_assignment = None
np.random.seed(100)
def Readdata():
   # Upload the dataset from local machine.
   uploaded = files.upload()
   myData = pd.read_csv(io.BytesIO(uploaded['LightDataset.csv']))
   #myData = pd.read_csv("/home/pi/.node-red/LightDataset.csv")
   myData.columns = ['motion', 'locationh', 'locationw', 'time', 'day', 'light']
   return myData

def dataprocessing(myData):
   myData = pd.get_dummies(myData, drop_first=True)
   myData = myData.dropna()
   x_train, x_test, y_train, y_test = train_test_split(myData.drop('light', axis=1),
                                                    myData['light'], test_size=.4, random_state=0, stratify=myData['light'])
   return x_train, x_test, y_train, y_test

def lstm(x_train, x_test, y_train, y_test):
  n_steps = 5
  n_features = 1
  x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1], n_features))
  x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], n_features))
  # define model
  model = Sequential()
  model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  # fit model
  model.fit(x_train, y_train, epochs=10, verbose=0, batch_size=5, validation_split=0.2)
  scores = model.evaluate(x_train, y_train, verbose=0)

  # serialize model to JSON
  model_json = model.to_json()
  with open("lstm_model_light.json", "w") as json_file:
    json_file.write(model_json)
    # serialize weights to HDF5
  model.save_weights("lstm_model_light.h5")
  print("Saved model to disk")

  # later...
  # load json and create model
  json_file = open('lstm_model_light.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)

  # load weights into new model
  loaded_model.load_weights("lstm_model_light.h5")
  print("Loaded model from disk")

  # evaluate loaded model on test data
  loaded_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
  score = loaded_model.evaluate(x_train, y_train, verbose=0)
  print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

if __name__ == "__main__":
  ''' call the functions '''
  data = Readdata()
  x_train, x_test, y_train, y_test = dataprocessing(data)
  lstm(x_train, x_test, y_train, y_test)