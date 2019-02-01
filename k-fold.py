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
from sklearn.model_selection import StratifiedKFold

#Load Data
data = pd.read_csv( 'HTRU_2.csv' )
# data.head()
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

print(y.shape[0])

# print("aaa ",y[0:4].dot(X[0:4,1:3]))

for i in range(1,9):
	X[:,i]=(X[:,i]-X[:,i].mean())/X[:,i].std()

def sigmoid(z):
	return 1/(1+np.exp(-z))

def costFunction(X,y,theta,_lambda=0):
	m=y.shape[0]
	h=sigmoid(X.dot(theta.T))
	reg=(_lambda/(2*m))*np.sum(theta**2)

	return (-1/m)*(y.dot(np.log(h))+(1-y).dot(np.log(1-h)))+reg

def der_costFunction(X,y,theta,_lambda=0):
	m=y.shape[0]
	y=y.reshape((m,1))
	# print("y ",y.shape)
	# print("ss ",X.shape,theta.shape)
	c=theta.T
	# print("c ",c.shape)
	h=sigmoid(X.dot(c))
	# print("h ",h.shape)
	b=h-y
	# print("b ",b.shape)
	a=(X.T.dot(b))
	# print("aaas ",h.shape,a.shape)
	reg=(_lambda*np.sum(theta))/m
	return (1/m)*a+reg

def gradientDescent(theta,X,y,lr=0.1,conv=0.00001):
	cost_iter=[]
	# print("aaa ",theta.shape)
	cost=costFunction(X,y,theta)
 	
	cost_iter.append([0 ,cost])
	change_cost=cost
	# print("aaa ",change_cost.shape)
	i=1
	while(change_cost>conv):
		# print("a")
		old_cost=cost
		theta=theta-(lr*der_costFunction(X,y,theta)).T
		# print("aaa ",theta.shape)
		cost=costFunction(X,y,theta)
		# print("aaa ",cost.shape)
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

folds = StratifiedKFold(n_splits=3)

scores_logistic = []
for train_index, test_index in folds.split(X,y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    theta = np.zeros((1,n+1))
    theta,cost_itr=gradientDescent(theta,X_train,y_train)
    predictions=predict(X_test,theta)
    scores_logistic.append(accuracy_score(y_test,predictions))

print(scores_logistic)

print(np.mean(scores_logistic))
