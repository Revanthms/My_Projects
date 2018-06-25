#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 19:56:14 2018

@author: revanth
"""

import pandas as pd
import numpy as np
import matplotlib as plt
#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
   
  Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print "Accuracy : %s" % "{0:.3%}".format(accuracy)

  Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print "Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 
  
  
df=pd.read_csv('/home/revanth/Documents/AV_projects/LoanPrediction/train_data.csv') #load training data


#temp1 = df['Credit_History'].value_counts(ascending=True)
#temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
#print 'Frequency Table for Credit History:' 
#print temp1
#
#print '\nProbility of getting loan for each Credit History class:' 
#print temp2
#
#temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
#temp3.plot(kind='bar', stacked=False, color=['green','blue'], grid=False)

df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)  #replace all the null values with mean
df['Self_Employed'].fillna('No',inplace=True) #replace all the null values with mode
df['Credit_History'].fillna(1,inplace=True) #replace all the null values with mode

#table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
# Define function to return value of this pivot_table
#def fage(x):
# return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values


df['LoanAmount'].fillna(360, inplace=True)
df['LoanAmount_log'] = np.log(df['LoanAmount'])


#df['LoanAmount_log'].hist(bins=20)

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])

#df['LoanAmount_log'].hist(bins=20) 


var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])




#Generic function for making a classification model and accessing performance:

  
outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History']
predictions = classification_model(model, df,predictor_var,outcome_var)


model = RandomForestClassifier(n_estimators=100) #we find this overfitting the training data. Play on with the hyperparameters to get the best cross validation accuracy
predictor_var = ['Gender', 'Married', 'Dependents', 'Education','Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area','LoanAmount_log','TotalIncome_log']
classification_model(model, df,predictor_var,outcome_var)


#Create a series with feature importances:
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print featimp

