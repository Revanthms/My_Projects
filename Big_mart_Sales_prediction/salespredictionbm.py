import cv2
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import sys
import math

df = pd.read_csv("/home/revanth/Documents/My_Projects/Big_mart_Sales_prediction/TrainSales.csv")
df['Item_Weight'].fillna(df['Item_Weight'].mean(),inplace=True)
df['Outlet_Size'].fillna('Medium',inplace=True) 
for i in df['Outlet_Establishment_Year']:
	i=2018-i
    

a=df['Item_Visibility']==0
for i in range(len(a)):                                        
     if a[i]==1:                    
            df['Item_Visibility'][i]=df['Item_Visibility'].mean()
#df['Item_Fat_Content'].value_counts().plot(kind='bar') 
for i in range(len(df['Item_Fat_Content'])):
     if df['Item_Fat_Content'][i]=='LF':
        df['Item_Fat_Content'][i]='Low Fat'
     elif df['Item_Fat_Content'][i]=='low fat':
         df['Item_Fat_Content'][i]='Low Fat'
     elif df['Item_Fat_Content'][i]=='reg':
         df['Item_Fat_Content'][i]='Regular'
var=['Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type']
le=LabelEncoder()
for i in var:
	df[i]=le.fit_transform(df[i])
matrixmode=df.as_matrix()
np.random.shuffle(matrixmode)
mat=np.delete(matrixmode,(0,11),1)
mat=mat.astype(float)
train_x= mat[:6000,]
test_x = mat[6000:8524,]
train_x=np.asmatrix(train_x)
test_x=np.asmatrix(test_x)


mat=np.delete(matrixmode,(0,1,2,3,4,5,6,7,8,9,10),1)
mat=mat.astype(float)
train_y=mat[:6000,]
test_y =mat[6000:8524,]
train_y=np.asmatrix(train_y)
test_y=np.asmatrix(test_y)

# first we will solve by analytics method

xtx=np.matmul(train_x.transpose(),train_x)
xtxi=np.linalg.pinv(xtx)
xtxixt=np.matmul(xtxi,train_x.transpose())
theta=np.matmul(xtxixt,train_y)

train_predicted=np.matmul(theta.transpose(),train_x.transpose())
train_predicted=train_predicted.transpose()

train_mse=metrics.mean_squared_error(train_predicted,train_y)
train_rmse=math.sqrt(train_mse)

test_predicted=np.matmul(theta.transpose(),test_x.transpose())
test_predicted=test_predicted.transpose()
test_mse=metrics.mean_squared_error(test_predicted,test_y)
test_rmse=math.sqrt(test_mse)


#now we will solve by gradient descent


class GradDescent:

    def __init__(self):
        self.deltaForDeriv = 0.001
        
        self.thetaBegin = np.matrix([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]).transpose()
        self.alpha = 0.1
        self.x = train_x
        self.ones = np.ones((6000, 1))
        self.xWithOnes = np.concatenate((self.x, self.ones), axis =1).transpose()
        self.xWithOnes = np.matrix(self.xWithOnes)
        self.yActual = train_y
	self.grad=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    def ErrorCalc(self, theta):
        self._y  = np.matmul(theta.transpose(), self.xWithOnes.transpose())
        
	self.train_mse=metrics.mean_squared_error(self._y.transpose(),self.yActual)                                                          
        return self._error

    def Deriv(self, theta):
        thetaPlus = theta
	for i in range(10):
        	thetaPlus[i] = theta[i] + self.deltaForDeriv
        	self._errorPlus_m_plus = self.ErrorCalc(thetaPlus)
        	thetaPlus[i] = theta[i] - self.deltaForDeriv
        	self._errorPlus_m_minus= self.ErrorCalc(thetaPlus)
        	self.grad[i] =float(self._errorPlus_m_plus-self._errorPlus_m_minus)/(2*self.deltaForDeriv)

       
	for i in range(10):
        	self.temp[i] = self.thetaBegin[i] - self.alpha*self.grad[i]
        for i in range(10):	
        	self.thetaBegin[i] = self.temp[i]
        
        #print self.thetaBegin.transpose()               

                            
        #return self._errorPluls

    def epochs(self, iter):
        for i in range(iter):
            self.Deriv(self.thetaBegin)
        print self.thetaBegin.transpose()
self = GradDescent()    
self.epochs(10000)



