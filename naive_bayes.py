
"""This naive bayes program will be using  all features of the date"""

import pandas as pd
from sklearn.model_selection import train_test_split

#read data
df = pd.read_excel('all_feat_clean_date.xlsx')

x = df.iloc[:, :34] # x-all features of the date
y = df.iloc[:, 34] # y-date class

"""Split data"""
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.3, random_state=0)

"""Standardize data"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

"""import the naive bayes model"""
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

y_predicted = naive_bayes.predict(x_test)
from sklearn.metrics import accuracy_score
y_pred_train = naive_bayes.predict(x_train)
print("Training Accuracy :" , accuracy_score(y_train, y_pred_train)*100)
print("Testing Accuracy :" , accuracy_score(y_test, y_predicted)*100)

#visualize the confusion matrix
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, accuracy_score 
import numpy as np 
cm  = confusion_matrix(y_test, y_predicted) 
cm_df = pd.DataFrame(cm,index = np.unique(y), columns =np.unique(y)) 
plt.subplots(figsize=(20,15)) 
sns.heatmap(cm_df, annot=True, 
cmap="YlGnBu",fmt=",",annot_kws={"size": 13}) 
plt.title('Naive Bayes Algorithm all feat \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_predicted))) 
plt.ylabel('y_test') 
plt.xlabel('y_test_pred') 
plt.show() 

#classification report
from sklearn.metrics import classification_report
grid_predictions = naive_bayes.predict(x_test)
#print classification report
print(classification_report(y_test, grid_predictions))

"""Backward elimination to find important features"""
import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


backward_best=sfs(estimator = naive_bayes, k_features ='best', forward = False ,verbose = 1, scoring='r2')
sfs_backward_best=backward_best.fit(x,y)

print('R-Squared value:', sfs_backward_best.k_score_)
feat_name = list(sfs_backward_best.k_feature_names_)
print(feat_name)



