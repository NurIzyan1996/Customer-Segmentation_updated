# -*- coding: utf-8 -*-
"""
Created on Fri May 20 00:51:51 2022

@author: USER
"""

import os
import pandas as pd
import numpy as np
import pickle
import missingno as msno
from tensorflow.keras.models import load_model
from modules import ExploratoryDataAnalysis

#%% PATH
DATA_PATH = os.path.join(os.getcwd(), 'datasets', 'new_customers.csv')
MARRIED_PATH = os.path.join(os.getcwd(), 'saved_model','Married_label.pkl')
GRADUATED_PATH = os.path.join(os.getcwd(), 'saved_model','Graduated_label.pkl')
PROFE_PATH = os.path.join(os.getcwd(), 'saved_model','Profession_label.pkl')
SPSCORE_PATH = os.path.join(os.getcwd(), 'saved_model','SpScore_label.pkl')
SEGMENT_PATH = os.path.join(os.getcwd(), 'saved_model','Segment_label.pkl')
KNN_PATH = os.path.join(os.getcwd(), 'saved_model','KNN_Imputer.pkl')
MMS_PATH = os.path.join(os.getcwd(), 'saved_model','mms_scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_model', 'model.h5')
SAVE_RESULT = os.path.join(os.getcwd(), 'datasets', 'new_customers_result.csv')
#%% STEP 1: Model Loading

# model Loading
mms_scaler = pickle.load(open(MMS_PATH,'rb'))
knn_scaler = pickle.load(open(KNN_PATH,'rb'))
segment_scaler = pickle.load(open(SEGMENT_PATH,'rb'))
spscore_scaler = pickle.load(open(SPSCORE_PATH,'rb'))
profe_scaler = pickle.load(open(PROFE_PATH,'rb'))
graduated_scaler = pickle.load(open(GRADUATED_PATH,'rb'))
married_scaler = pickle.load(open(MARRIED_PATH,'rb'))

model = load_model(MODEL_PATH)
model.summary()

#%% STEP 2: Data Loading
df = pd.read_csv(DATA_PATH)

# STEP 3: Data Inspection
# a) to display the first 10 rows of data
print(df.head(10))

# b) to view the summary, non-null
print(df.info())
'''Observation: Ever_Married, Graduated, Profession, Work_Experience, 
Family_Size, Var_1 contain Null Values'''

# c) split features and target
target = df['Segmentation'] # target data

# features data
num_data = df[['Age', 'Work_Experience', 'Family_Size']]
cat_data = df[['Gender','Ever_Married', 'Graduated', 'Profession', 
            'Spending_Score', 'Var_1']]

# d) checking NaN values using missingno approach
msno.matrix(df)
''' Observation: column 'Work_Experience' has a lot of null values.'''

#%% STEP 3: Data Cleaning

# a) eliminate Gender, Var_1, Work_Experience to deploy label scaler
num_data = df[['Age', 'Family_Size']]
cat_data = df[['Ever_Married', 'Graduated', 'Profession', 'Spending_Score']]

# b) encode categorical data using Labe Encoder approach
eda = ExploratoryDataAnalysis()
eda.label_transform(cat_data['Ever_Married'],married_scaler)
eda.label_transform(cat_data['Graduated'],graduated_scaler)
eda.label_transform(cat_data['Profession'],profe_scaler)
eda.label_transform(cat_data['Spending_Score'],spscore_scaler)

# c) fill nan using KNNImputer
features = pd.concat([cat_data, num_data], axis=1)
features = eda.knn_imputer_transform(features,knn_scaler)

#%% STEP 4: Data Preprocessing

# a) scale the features using Min Max Scaler approach
features = mms_scaler.transform(features)

#%% STEP 5: Deployment
predicted_y = np.empty([len(features), 4])

for index, test in enumerate(features):
    predicted_y[index,:] = model.predict(np.expand_dims(test, axis=0))

#%% STEP 6: Data Update 
df['Segmentation'] = segment_scaler.inverse_transform(np.argmax(predicted_y, 
                                                                axis=1))

df.to_csv(SAVE_RESULT, index=False)


