""" 
TrainFlowData.py
Sonia Gu '22 
Trains an AutoGluon-Tabular model. The training data should be
in the format where every row contains the information for 
one trace and the first n - 1 columns contain the processed 
metadata, and the nth column contains the trace's label. 

The predictor information is stored as a folder called save_path
"""

from autogluon.tabular import TabularDataset, TabularPredictor
import sys

train_data_csv = sys.argv[1] # the processed training data in csv format
subsample_size = sys.argv[2] # size of the sample to train over. Must be less than or equal to the size of the training data
save_path = sys.argv[3]  # specifies folder to store trained models

train_data = TabularDataset(train_data_csv)
train_data.fillna(0, inplace=True)

train_data = train_data.sample(n = int(subsample_size), random_state=0)
print(train_data.head())

label = 'label'
print("Summary of class variable: \n", train_data[label].describe())

predictor = TabularPredictor(label=label, path=save_path).fit(train_data, time_limit=83000).fit_summary(3, True)
