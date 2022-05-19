# -*- coding: utf-8 -*-
"""
Created on Thu May 19 20:46:10 2022

@author: Nur Izyan Binti Kamarudin
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout,BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.metrics import accuracy_score

class DataVisualization():
    
    def __init__(self):
        pass
    
    def bar_plot(self, input_data, hue):
        fig , ax = plt.subplots(3,2,figsize = (15,11))
        for i , subplots in zip (input_data , ax.flatten()):
          sns.countplot(input_data[i], hue=hue, ax = subplots)
        return plt.show()
    
class ExploratoryDataAnalysis():
    def __init__(self):
        pass
    
    def label_encoder(self, data,path):
        le = LabelEncoder()
        data[data.notnull()] = le.fit_transform(data[data.notnull()])
        pickle.dump(le, open(path, 'wb'))
        return data
    
    def label_transform(self, data, scaler):
        data[data.notnull()] = scaler.transform(data[data.notnull()])
        return data
    
    def knn_imputer(self, data,path):
        imputer = KNNImputer(n_neighbors=5, metric='nan_euclidean') 
        data = pd.DataFrame(imputer.fit_transform(data))
        pickle.dump(imputer, open(path, 'wb'))
        return data
    
    def knn_imputer_transform(self, data, scaler):
        data = pd.DataFrame(scaler.transform(data))
        return data
    
    
class DataPreprocessing():
    
    def __init__(self):
        pass
    
    def min_max_scaler(self, data, path):
        mms = MinMaxScaler()
        data = mms.fit_transform(data)
        pickle.dump(mms, open(path, 'wb'))
        return data
        
    def one_hot_encoder(self, data,path):  
        enc = OneHotEncoder(sparse=False) 
        data = enc.fit_transform(np.expand_dims(data,axis=-1))
        pickle.dump(enc, open(path, 'wb'))
        return data

class ModelCreation():
    def __init__(self):
        pass
    
    def sequential(self, input_shape, output_shape, nb_nodes):
        model = Sequential()
        model.add(Input(shape=(input_shape), name='input_layer'))
        model.add(Flatten()) # to flatten the data
        model.add(Dense(nb_nodes, activation='relu', name='hidden_layer_1'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(nb_nodes, activation='relu', name='hidden_layer_2'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(nb_nodes, activation='relu', name='hidden_layer_3'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(nb_nodes, activation='relu', name='hidden_layer_4'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(output_shape, activation='softmax', name='output_layer'))
        model.summary()
        return model

class ModelEvaluation():
    def report_metrics(self,y_true,y_pred):
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        print((accuracy_score(y_true, y_pred))*100,'%')