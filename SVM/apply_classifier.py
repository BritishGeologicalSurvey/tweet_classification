#Script for applying SVM Linear Regression model
# "1" means "aurora-event", "0" means "no aurora-event"
# Usage: python apply_classifier.py 

import pickle
import csv
import json
import os
import numpy as np
from classification_module import NLTKPreprocessor , identity 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import pdb
import re
import codecs
import operator  
from sklearn.metrics import classification_report as clsr
from sklearn.metrics import accuracy_score as acc

# This script reads the test files line by line and creates one label array and one tweet array. It then fits the trained model to the data.
array = []
label = []
#Load trained model
model_dir = "./"
model_file = "model.pickle"
model_name =  model_dir + model_file
with open(model_name,'rb') as f:
    model=pickle.load(f)


#Load test data
filenames = ['test_BGS.neg','test_BGS.pos']
with open(filenames[0],"r") as pos, open(filenames[1]) as neg:
        for line in pos:
            array.append(line)
            label.append(1)
        for line in neg:
            label.append(0)
            array.append(line)
labels = LabelEncoder()
y = labels.fit_transform(label)
                                         
y_pred = model.predict(array)
tot=0
np.asarray(label)
for l in range(len(y_pred)):
    if y_pred[l]!=label[l]:
        tot=tot+1
print(clsr(label,y_pred))
accuracy = acc(label,y_pred)
print("Accuracy:{}".format(accuracy))
