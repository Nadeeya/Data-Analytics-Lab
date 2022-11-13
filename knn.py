# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 16:41:37 2022

@author: Nadeeya Norjemee
knn - catagorical

"""

import numpy as np
import pandas as pd

df = pd.read_excel('Date1.xlsx')

x = df.iloc[:, :34]
y = df.iloc[:, 34]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=9,  metric='manhattan', weights='distance')
clf.fit(X_train, y_train)


# generate evaluation metrics

from sklearn.metrics import accuracy_score
y_pred = clf.predict(X_train)
print("Training Accuracy :" , accuracy_score(y_train, y_pred)*100)

y_pred_test = clf.predict(X_test)
print("Testing Accuracy :" , accuracy_score(y_test, y_pred_test)*100)

y_pred = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, accuracy_score 
cm  = confusion_matrix(y_test, y_pred_test) 
cm_df = pd.DataFrame(cm,index = np.unique(y), columns =np.unique(y)) 
plt.subplots(figsize=(20,15)) 
sns.heatmap(cm_df, annot=True, 
cmap="YlGnBu",fmt=",",annot_kws={"size": 13}) 
plt.title('KNN Algorithm \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred_test))) 
plt.ylabel('y_test') 
plt.xlabel('y_test_pred') 
plt.show() 


from sklearn.metrics import classification_report
grid_predictions = clf.predict(X_test)

#print classification report
print(classification_report(y_test, grid_predictions))