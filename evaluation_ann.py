# Evaluation ANN modeles (MLPNN, LSTM, and GRU) to select the best for our system.

import numpy as np
import pandas as pd
import io
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from google.colab import files
from keras.models import Sequential
from keras.layers import Dense, ReLU, LSTM, GRU
import tensorflow as tf  # To remove logging

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
pd.options.mode.chained_assignment = None
np.random.seed(100)


def Readdata():
    # Upload the dataset from local machine.
    uploaded = files.upload()
    myData = pd.read_csv(io.BytesIO(uploaded['LightDataset.csv']))
    # myData = pd.read_csv("/home/pi/.node-red/LightDataset.csv")
    myData.columns = ['motion', 'locationh', 'locationw', 'time', 'day', 'light']
    return myData


def dataprocessing(myData):
    # data processing
    myData = pd.get_dummies(myData, drop_first=True)
    myData = myData.dropna()
    x_train, x_test, y_train, y_test = train_test_split(myData.drop('light', axis=1),
                                                        myData['light'], test_size=.4, random_state=0,
                                                        stratify=myData['light'])
    return x_train, x_test, y_train, y_test


# Build MLPNN
def MLPNN(x_train, x_test, y_train, y_test):
    nn_model = Sequential()
    nn_model.add(Dense(9, input_shape=(5,), activation='sigmoid'))
    nn_model.add(Dense(1, activation='sigmoid'))
    nn_model.compile(optimizer='Adam', loss='binary_crossentropy')
    nn_model.fit(x_train, y_train, epochs=10, verbose=0, batch_size=5, validation_split=0.2)
    y_predicted = (nn_model.predict(x_test) > 0.5)
    conf_mat = confusion_matrix(y_test, y_predicted)
    print(conf_mat)
    total = sum(sum(conf_mat))
    sensitivity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[1, 0])
    specificity = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])
    accuracy = (conf_mat[0, 0] + conf_mat[1, 1]) / total
    Precision = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[1, 0])
    Recall = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])
    F1 = 2 * Precision * Recall / (Precision + Recall)
    fpr_mlpnn, tpr_mlpnn, thresholds_mlpnn = roc_curve(y_test, y_predicted)
    auc_mlpnn = auc(fpr_mlpnn, tpr_mlpnn)
    print('MLPNN confusion matrix : ', conf_mat)
    print('MLPNN specificity : ', specificity)
    print('MLPNN sensitivity : ', sensitivity)
    print('MLPNN accuracy : ', accuracy)
    print('MLPNN Precision : ', Precision)
    print('MLPNN Recall : ', Recall)
    print('MLPNN F1-Score : ', F1)
    return fpr_mlpnn, tpr_mlpnn, thresholds_mlpnn, auc_mlpnn


# build GRU
def gru(x_train, x_test, y_train, y_test):
    n_steps = 5
    n_features = 1
    x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1], n_features))
    x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(GRU(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    # fit model
    model.fit(x_train, y_train, epochs=10, verbose=0, batch_size=5, validation_split=0.2)
    y_predicted = (model.predict(x_test) > 0.5)
    conf_mat = confusion_matrix(y_test, y_predicted)
    total = sum(sum(conf_mat))
    sensitivity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[1, 0])
    specificity = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])
    accuracy = (conf_mat[0, 0] + conf_mat[1, 1]) / total
    Precision = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[1, 0])
    Recall = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])
    F1 = 2 * Precision * Recall / (Precision + Recall)
    fpr_gru, tpr_gru, thresholds_gru = roc_curve(y_test, y_predicted)
    auc_gru = auc(fpr_gru, tpr_gru)
    print('GRUs confusion matrix : ', conf_mat)
    print('GRUs specificity : ', specificity)
    print('GRUs sensitivity : ', sensitivity)
    print('GRUs accuracy : ', accuracy)
    print('GRUs Precision : ', Precision)
    print('GRUs Recall : ', Recall)
    print('GRUs F1-Score : ', F1)
    return fpr_gru, tpr_gru, thresholds_gru, auc_gru


# build LSTM
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
    y_predicted = (model.predict(x_test) > 0.5)
    conf_mat = confusion_matrix(y_test, y_predicted)
    total = sum(sum(conf_mat))
    sensitivity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[1, 0])
    specificity = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])
    accuracy = (conf_mat[0, 0] + conf_mat[1, 1]) / total
    Precision = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[1, 0])
    Recall = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])
    F1 = 2 * Precision * Recall / (Precision + Recall)
    fpr_lstm, tpr_lstm, thresholds_lstm = roc_curve(y_test, y_predicted)
    auc_lstm = auc(fpr_lstm, tpr_lstm)
    print('LSTM confusion matrix:', conf_mat)
    print('LSTM specificity : ', specificity)
    print('LSTM sensitivity : ', sensitivity)
    print('LSTM accuracy : ', accuracy)
    print('LSTM Precision : ', Precision)
    print('LSTM Recall : ', Recall)
    print('LSTM F1-Score : ', F1)
    return fpr_lstm, tpr_lstm, thresholds_lstm, auc_lstm


# Build the ROC the show the relation between FPR and TPR
def roc_(fpr_mlpnn, tpr_mlpnn, thresholds_mlpnn, auc_mlpnn, fpr_gru, tpr_gru,
         thresholds_gru, auc_gru, fpr_lstm, tpr_lstm, thresholds_lstm, auc_lstm):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_mlpnn, tpr_mlpnn, label='MLPN (area = {:.3f})'.format(auc_mlpnn))
    plt.plot(fpr_gru, tpr_gru, label='GRUs (area = {:.3f})'.format(auc_gru))
    plt.plot(fpr_lstm, tpr_lstm, label='LSTM (area = {:.3f})'.format(auc_lstm))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


# main function
if __name__ == "__main__":
    ''' call the functions '''
    data = Readdata()
    x_train, x_test, y_train, y_test = dataprocessing(data)
    fpr_mlpnn, tpr_mlpnn, thresholds_mlpnn, auc_mlpnn = MLPNN(x_train, x_test, y_train, y_test)
    fpr_gru, tpr_gru, thresholds_gru, auc_gru = gru(x_train, x_test, y_train, y_test)
    fpr_lstm, tpr_lstm, thresholds_lstm, auc_lstm = lstm(x_train, x_test, y_train, y_test)
    roc_(fpr_mlpnn, tpr_mlpnn, thresholds_mlpnn, auc_mlpnn, fpr_gru, tpr_gru,
         thresholds_gru, auc_gru, fpr_lstm, tpr_lstm, thresholds_lstm, auc_lstm)