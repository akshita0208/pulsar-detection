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
from numpy.random import rand

data = pd.read_csv( 'HTRU_2.csv' )
# data.head()
m=data.shape[0]
n=8
k=2
X=np.ones((m,n))
y=np.ones((m,1))
X[:,0]=data['x1'].values
X[:,1]=data['x2'].values
X[:,2]=data['x3'].values
X[:,3]=data['x4'].values
X[:,4]=data['x5'].values
X[:,5]=data['x6'].values
X[:,6]=data['x7'].values
X[:,7]=data['x8'].values
y=data['y'].values

for i in range(8):
	X[:,i]=(X[:,i]-X[:,i].mean())/X[:,i].std()



def sigmoid(z):
	return 1/(1+np.exp(-z))

def sigmoid_der(z):
	return (np.exp(z)/((1+np.exp(-z))**2))

def ReLU(x):
    return x * (x > 0)

def ReLU_der(x):
    return 1. * (x > 0)

def tanh(z):
	a=np.exp(z)
	b=np.exp(-z)
	return (a-b)/(a+b)

def tanh_der(z):
	a=np.exp(z)
	b=np.exp(-z)
	c=(a-b)/(a+b)
	return (1-c**2)



def Forward_prop(X,Model):
	w1,b1,w2,b2,w3,b3=Model['w1'], Model['b1'], Model['w2'], Model['b2'], Model['w3'],Model['b3']
	# print("e")
	# print(w1.shape)
	# print(b1.shape)
	# print(w2.shape)
	# print(b2.shape)
	# print(w3.shape)
	# print(b3.shape)
	z1=w1.dot(X)+b1
	a1=tanh(z1)
	z2=w2.dot(a1)+b2
	a2=tanh(z2)
	z3=w3.dot(a2)+b3
	a3=sigmoid(z3)
	cache = {'a0':X,'z1':z1,'a1':a1,'z2':z2,'a2':a2,'a3':a3,'z3':z3}
	return cache

def Back_prop(y,Model,cache):
	w1,b1,w2,b2,w3,b3=Model['w1'], Model['b1'], Model['w2'], Model['b2'], Model['w3'],Model['b3']
	a0,a1, a2,a3,z1,z2,z3 = cache['a0'],cache['a1'],cache['a2'],cache['a3'],cache['z1'],cache['z2'],cache['z3']
	dz3=a3-y
	dw3=dz3.dot(a2.T)
	db3=np.sum(dz3,axis=1)
	dz2=np.multiply(w3.T.dot(dz3),tanh_der(z2))
	dw2=dz2.dot(a1.T)
	db2=np.sum(dz2,axis=1)
	dz1=np.multiply(w2.T.dot(dz2),tanh_der(z1))
	dw1=dz1.dot(a0.T)
	db1=np.sum(dz1,axis=1)
	# print("g")
	# print(dw1.shape)
	# print(db1.shape)
	# print(dw2.shape)
	# print(db2.shape)
	# print(dw3.shape)
	# print(db3.shape)

	grads = {'dW3':dw3, 'db3':db3, 'dW2':dw2,'db2':db2,'dW1':dw1,'db1':db1}
	return grads

def Gradient_desc(X,y,Model,lr=6.5):
	cost_iter=[]
	m=y.shape[1]
	cost =Forward_prop(X,Model)
	cost_iter.append([0 ,cost['a3']])
	
	for i in range(1,50):
		grads=Back_prop(y,Model,cost)
		w1,b1,w2,b2,w3,b3=Model['w1'], Model['b1'], Model['w2'], Model['b2'], Model['w3'],Model['b3']
		dw1,db1,dw2,db2,dw3,db3=grads['dW1'],grads['db1'],grads['dW2'],grads['db2'],grads['dW3'],grads['db3']
		w1=w1-lr*(dw1/m)
		w2=w2-lr*(dw2/m)
		w3=w3-lr*(dw3/m)
		b1=b1-lr*(db1.reshape(10,1)/m)
		b2=b2-lr*(db2.reshape(5,1)/m)
		b3=b3-lr*(db3.reshape(1,1)/m)
		Model = {'w1':w1,'b1':b1,'w2':w2,'b2':b2,'w3':w3,'b3':b3}
		cost =Forward_prop(X,Model)
		cost_iter.append([i ,cost['a3']])

	return Model,cost_iter


def predict(X,Model,hard=True):
	cost=Forward_prop(X,Model)
	a3=cost['a3']
	value=np.where(a3>=0.5 ,1, 0)
	if hard:
		return value
	return a3


X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size = 0.4, random_state = 11)

X_rem_cv, X_rem_test, y_rem_cv, y_rem_test = train_test_split(X_rem, y_rem, test_size = 0.5, random_state = 11)

X_train= X_train.T
y_train=y_train.reshape((y_train.shape[0],1))
y_train=y_train.T
X_rem_cv= X_rem_cv.T
y_rem_cv=y_rem_cv.reshape((y_rem_cv.shape[0],1))
y_rem_cv=y_rem_cv.T
X_rem_test= X_rem_test.T
y_rem_test=y_rem_test.reshape((y_rem_test.shape[0],1))
y_rem_test=y_rem_test.T

print(X_train.shape)
#Hidden Layers = 3
#Hidden units in layer 1 =10
#Hidden units in layer 2 =5

L=3
n1=10
n2=5
#random Initialisation
w1=rand(n1,n)*0.01
b1=np.zeros((n1,1))
w2=rand(n2,n1)*0.01
b2=np.zeros((n2,1))
w3=rand(1,n2)*0.01
b3=np.zeros((1,1))
print(w1.shape)
print(b1.shape)
print(w2.shape)
print(b2.shape)
print(w3.shape)
print(b3.shape)

Model = {'w1':w1,'b1':b1,'w2':w2,'b2':b2,'w3':w3,'b3':b3}

Model,cost_iter=Gradient_desc(X_train,y_train,Model)
# print("aa")
predictions_test=predict(X_rem_test,Model)
predictions=predict(X_rem_cv,Model)
predictions_train=predict(X_train,Model)

print(classification_report(y_rem_cv.T,predictions.T))
print(accuracy_score(y_rem_test.T,predictions_test.T))


#to find cross validation and traing error

tmp_1=np.logical_not(predictions.T)
z_1=np.logical_and(tmp_1,y_rem_cv.reshape((y_rem_cv.shape[1],1)))
print(z_1.shape)
count=0
for i in range(z_1.shape[0]):
	if(z_1[i][0]!=0):
		count+=1

print("cv ",count*100/z_1.shape[0])
tmp_2=np.logical_not(predictions_train.T)
z_2=np.logical_and(tmp_2,y_train.reshape((y_train.shape[1],1)))
#print(z.shape)
count2=0
for i in range(z_2.shape[0]):
	if(z_2[i][0]!=0):
		count2+=1

print("train ",count2*100/z_2.shape[0])

#Costs

c=Forward_prop(X_train,Model)
t_a3=c['a3']
train_cost=np.sum((t_a3 - y_train)**2)/t_a3.shape[1]
# print(train_cost.shape)
c=Forward_prop(X_rem_cv,Model)
cv_a3=c['a3']
cv_cost=np.sum((cv_a3 - y_rem_cv)**2)/cv_a3.shape[1]
print("training cost: ",train_cost,"cross_validation cost: ",cv_cost)




	






