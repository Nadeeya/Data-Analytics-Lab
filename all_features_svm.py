# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:20:32 2022

@author: Nadeeya Norjemee
This svm model will use all features of the dates
"""
import pandas as pd
from sklearn.model_selection import train_test_split
#read data
df = pd.read_excel('Date1.xlsx')

x = df.iloc[:, :34] # x- all features of the date
y = df.iloc[:, 34] # y- date class

"""Splitting data"""
#Splitting data into training and testing 70:30 with random state = 0
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.3, random_state=0)

"""Standardize the data"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

"""Import model and fit with training data"""
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=10, gamma=1)
svm.fit(x_train, y_train)

"""Evaluating model performance"""

from sklearn.metrics import accuracy_score
y_pred = svm.predict(x_train)
print("Training Accuracy :" , accuracy_score(y_train, y_pred)*100)

y_pred_test = svm.predict(x_test)
print("Testing Accuracy :" , accuracy_score(y_test, y_pred_test)*100)

#visualize the confusion matrix
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, accuracy_score 
cm  = confusion_matrix(y_test, y_pred_test) 
cm_df = pd.DataFrame(cm,index = np.unique(y), columns =np.unique(y)) 
plt.subplots(figsize=(20,15)) 
sns.heatmap(cm_df, annot=True, 
cmap="YlGnBu",fmt=",",annot_kws={"size": 13}) 
plt.title('SVM Algorithm \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred_test))) 
plt.ylabel('y_test') 
plt.xlabel('y_test_pred') 
plt.show() 

#Classification report
from sklearn.metrics import classification_report
grid_predictions = svm.predict(x_test)
#print classification report
print(classification_report(y_test, grid_predictions))

"""Forward Feature selection to find important features"""
import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
forward_fs_best=sfs(estimator = svm, k_features ='best', forward = True,verbose = 1, scoring = 'r2')
sfs_forward_best=forward_fs_best.fit(x,y)
print('R-Squared value:', sfs_forward_best.k_score_)
feat_name = list(sfs_forward_best.k_feature_names_)
print(feat_name)


