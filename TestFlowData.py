""" 
TestFlowData.py
Sonia Gu '22 
Tests an AutoGluon-Tabular model. The test data should be
in the format where every row contains the information for 
one trace and the first n - 1 columns contain the processed 
metadata, and the nth column contains the trace's label. 

If the data is in the closed world setting, then this code 
prints the accuracy of each model. In the open world setting, 
this code prints the recall and precision. 
"""

from autogluon.tabular import TabularDataset, TabularPredictor
import sys
import sklearn as sk
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

test_data_csv = sys.argv[1] # csv of the testing data
save_path = sys.argv[2] # the path of the trained predictor
setting = sys.arg[3] # either "closed" or "open"
num_classes = sys.argv[4] # number of websites in the dataset. Optional if in the open world setting (the number of classes is always 2) 
F1_score_plot_name = sys.argv[5] # image file to save the figure of the F1 scores in. Optional if in the open world setting.

test_data = TabularDataset(test_data_csv)
label = "label"
y_test = test_data[label]  # values to predict
test_data_nolab = test_data.drop(columns=[label])  # delete label column to prove we're not cheating
print("save_path: ", save_path)
print(test_data_nolab.head())

predictor = TabularPredictor.load(save_path) 

y_pred = predictor.predict(test_data_nolab)
print("Predictions:  \n", y_pred)

extra_metrics = []

if setting == "closed":
    f1score = sk.metrics.f1_score(y_test, y_pred, average=None)

    print("F1_score (average=None): ", f1score)
    print("F1_score (average=macro): ", sk.metrics.f1_score(y_test, y_pred, average='macro'))

    plt.scatter(range(int(sys.argv[3])), f1score)
    plt.xlabel("Website label")
    plt.ylabel("F1 score")
    plt.title("F1 score of each website in the closed-world dataset")
    plt.savefig(F1_score_plot_name) 

elif setting == "open":
    print("confusion matrix:", sk.metrics.confusion_matrix(y_test, y_pred))

    print("recall", sk.metrics.recall_score(y_test, y_pred))
    print("precision", sk.metrics.precision_score(y_test, y_pred))
    
    extra_metrics = ['recall', 'precision']
    
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
print(perf)

pd.set_option('display.max_columns', None)
print(predictor.leaderboard(test_data, extra_metrics=extra_metrics, silent=True))
