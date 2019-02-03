	#!/usr/bin/env python3
from subprocess import check_output
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import warnings 
# warnings.filterwarnings('ignore')
from math import ceil
#Plots
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix #Confusion matrix
from sklearn.metrics import accuracy_score  
from sklearn.cross_validation import train_test_split
from pandas.tools.plotting import parallel_coordinates
#Advanced optimization
from scipy import optimize as op
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


#Load Data
data = pd.read_csv( 'HTRU_2.csv' )
m=data.shape[0]
n=8
k=2
X=np.ones((m,n+1))
y=np.ones((m,1))
X[:,1]=data['x1'].values
X[:,2]=data['x2'].values
X[:,3]=data['x3'].values
X[:,4]=data['x4'].values
X[:,5]=data['x5'].values
X[:,6]=data['x6'].values
X[:,7]=data['x7'].values
X[:,8]=data['x8'].values
y=data['y'].values

# print(y.shape[0])


for i in range(1,9):
	X[:,i]=(X[:,i]-X[:,i].mean())/X[:,i].std()

def sigmoid(z):
	return 1/(1+np.exp(-z))

def costFunction(X,y,theta,_lambda=0.1):
	m=y.shape[0]
	h=sigmoid(X.dot(theta.T))
	reg=(_lambda/(2*m))*np.sum(theta**2)

	return (-1/m)*(y.dot(np.log(h))+(1-y).dot(np.log(1-h)))+reg

def der_costFunction(X,y,theta,_lambda=0.1):
	m=y.shape[0]
	y=y.reshape((m,1))
	c=theta.T
	h=sigmoid(X.dot(c))
	b=h-y
	a=(X.T.dot(b))
	reg=(_lambda*np.sum(theta))/m
	return (1/m)*a+reg

def gradientDescent(theta,X,y,lr=0.001,conv=0.00001):
	cost_iter=[]
	cost=costFunction(X,y,theta)
	cost_iter.append([0 ,cost])
	change_cost=cost
	i=1
	while(change_cost>conv):
		old_cost=cost
		theta=theta-(lr*der_costFunction(X,y,theta)).T
		cost=costFunction(X,y,theta)
		cost_iter.append([i ,cost])
		change_cost=old_cost-cost
		i+=1
	return theta,np.array(cost_iter)

def predict(X,theta,hard=True):
	p=sigmoid(X.dot(theta.T))
	value=np.where(p>=0.5 ,1, 0)
	if hard:
		return value
	return p
'''class LogisticRegression:
    def __init__(self, lr=0.001, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
       
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
                
            if(self.verbose ==True and i % 10000 == 0):
                print(f'loss: {loss} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()
	'''



X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size = 0.4, random_state = 11)

X_rem_cv, X_rem_test, y_rem_cv, y_rem_test = train_test_split(X_rem, y_rem, test_size = 0.5, random_state = 11)

theta = np.zeros((1,n+1))
# print(costFunction(X_train,y_train,theta))
# print(X_train.shape , y_train.shape)
# alpha=der_costFunction(X_train,y_train,theta)
# print(alpha)
theta,cost_itr=gradientDescent(theta,X_train,y_train)

predictions=predict(X_rem_cv,theta)
print(theta)
print(classification_report(y_rem_cv,predictions))
print(accuracy_score(y_rem_cv,predictions))

print("training cost: ",costFunction(X_train,y_train,theta) ,"cross_validation cost: ",costFunction(X_rem_cv,y_rem_cv,theta))

# # h=sigmoid(X.dot(theta))
# # h=np.log(h)
# # h=y.T.dot(h)
# # print(h)
# print(X[0,0],X[0,1])
