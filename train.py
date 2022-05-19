# -*- coding: utf-8 -*-
"""
Created on Thu May 19 20:41:34 2022

@author: Nur Izyan Binti Kamarudin
"""

import os
import pandas as pd
import numpy as np
import missingno as msno
import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from modules import DataVisualization,ExploratoryDataAnalysis,DataPreprocessing
from modules import ModelCreation, ModelEvaluation
#%% Paths
DATA_PATH = os.path.join(os.getcwd(), 'datasets', 'train.csv')
LOG_PATH = os.path.join(os.getcwd(),'log')
MARRIED_PATH = os.path.join(os.getcwd(), 'saved_model','Married_label.pkl')
GRADUATED_PATH = os.path.join(os.getcwd(), 'saved_model','Graduated_label.pkl')
PROFE_PATH = os.path.join(os.getcwd(), 'saved_model','Profession_label.pkl')
SPSCORE_PATH = os.path.join(os.getcwd(), 'saved_model','SpScore_label.pkl')
SEGMENT_PATH = os.path.join(os.getcwd(), 'saved_model','Segment_label.pkl')
KNN_PATH = os.path.join(os.getcwd(), 'saved_model','KNN_Imputer.pkl')
MMS_PATH = os.path.join(os.getcwd(), 'saved_model','mms_scaler.pkl')
OHE_PATH = os.path.join(os.getcwd(), 'saved_model','ohe_scaler.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model', 'model.h5')
#%% EDA
# STEP 1: Data Loading
df = pd.read_csv(DATA_PATH)

# STEP 2: Data Inspection
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

# d) find trends of features and target data
dv = DataVisualization()
dv.bar_plot(cat_data,target)
'''Observation: The barplot_trends.png shows that column 'Gender' and 'Var_1' 
shows a weak trends. Eliminating these columns will helps training a model
efficiently and accurately'''

# e)checking NaN values using missingno approach
msno.matrix(df)
''' Observation: column 'Work_Experience' has a lot of null values. Eliminating 
these columns will helps training a model efficiently and accurately'''

#%% STEP 3: Data Cleaning

# a) eliminate Gender, Var_1, Work_Experience
num_data = df[['Age', 'Family_Size']]
cat_data = df[['Ever_Married', 'Graduated', 'Profession', 'Spending_Score']]

# b) encode categorical data using Labe Encoder approach
eda = ExploratoryDataAnalysis()
eda.label_encoder(cat_data['Ever_Married'], MARRIED_PATH)
eda.label_encoder(cat_data['Graduated'], GRADUATED_PATH)
eda.label_encoder(cat_data['Profession'], PROFE_PATH)
eda.label_encoder(cat_data['Spending_Score'], SPSCORE_PATH)
target = eda.label_encoder(target, SEGMENT_PATH)

# c) fill nan using KNNImputer
features = pd.concat([cat_data, num_data], axis=1)
new_df = pd.concat([features, target], axis=1)
new_df = eda.knn_imputer(new_df, KNN_PATH)

# d) checking Null values
new_df.info()
''' Observation: there is no null value'''

#%% STEP 4: Data Preprocessing

X = new_df.iloc[:,0:6] # features
y = new_df.iloc[:,6] # target
# a) scale the features using Min Max Scaler approach
dp = DataPreprocessing()
X = dp.min_max_scaler(X, MMS_PATH)

# b) encode the target data using One Hot encoder approach
y = dp.one_hot_encoder(y, OHE_PATH)

#%%  STEP 5: DL Model
# a) split train & test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42)
X_train = np.expand_dims(X_train,-1)
X_test = np.expand_dims(X_test,-1)

# b) create model
mc = ModelCreation()
model = mc.sequential(input_shape=6, output_shape=4, nb_nodes=256)

plot_model(model)

model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['acc'])

# c) Callbacks 
log_files = os.path.join(LOG_PATH, 
                          datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=50)

# e) train the model
hist = model.fit(X_train, y_train, epochs=100, 
                    validation_data=(X_test,y_test), 
                    callbacks=[tensorboard_callback,early_stopping_callback])


#%% STEP 6: Model Evaluation

predicted_y = np.empty([len(X_test), 4])

for index, test in enumerate(X_test):
    predicted_y[index,:] = model.predict(np.expand_dims(test, axis=0))

#%% STEP 7: Model analysis
y_pred = np.argmax(predicted_y, axis=1)
y_true = np.argmax(y_test, axis=1)

ModelEvaluation().report_metrics(y_true, y_pred)

model.save(MODEL_SAVE_PATH)







