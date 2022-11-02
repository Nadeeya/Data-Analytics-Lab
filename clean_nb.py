# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 20:01:07 2022
@author: Nadeeya Norjemee
This naive bayes model is using only important features of the date
"""

import pandas as pd
from sklearn.model_selection import train_test_split

#read data
df = pd.read_excel('clean_date_nb.xlsx')

x = df.iloc[:, :11] # x - the 11 important features
y = df.iloc[:, 11] # y - the date class

"""Splitting data"""

#The dataset is split into 70:30 , train:test ratio with a random state of 0
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.3, random_state=0)

"""Standardizing data"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

"""import naive bayes model and fit the test data into the model"""

from sklearn.naive_bayes import GaussianNB
#Calling the Class
naive_bayes = GaussianNB()
#Fitting the data to the classifier
naive_bayes.fit(x_train , y_train) 
#Predict on test data
y_predicted = naive_bayes.predict(x_test)

"""Evaluating model performance"""

from sklearn import metrics 
metrics.accuracy_score(y_predicted , y_test)
from sklearn.metrics import accuracy_score
y_pred_train = naive_bayes.predict(x_train)
print("Training Accuracy :" , accuracy_score(y_train, y_pred_train)*100)
print("Testing Accuracy :" , accuracy_score(y_test, y_predicted)*100)

#Visualizing confusion matrix
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, accuracy_score 
import numpy as np
cm  = confusion_matrix(y_test, y_predicted) 
cm_df = pd.DataFrame(cm,index = np.unique(y), columns =np.unique(y)) 
plt.subplots(figsize=(20,15)) 
sns.heatmap(cm_df, annot=True, 
cmap="YlGnBu",fmt=",",annot_kws={"size": 13}) 
plt.title('Naive Bayes Algorithm \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_predicted))) 
plt.ylabel('y_test') 
plt.xlabel('y_test_pred') 
plt.show() 

#Classification Report
from sklearn.metrics import classification_report
grid_predictions = naive_bayes.predict(x_test)

#print classification report
print(classification_report(y_test, grid_predictions))