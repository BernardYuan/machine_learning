import numpy as np 
import scipy as scp 
import matplotlib.pyplot as plt 
import math

def data_generator1(N,d=48):
	# x is feature, y is label
	mean0 = np.zeros(d) 
	covar = np.identity(d)
	covar = np.matrix([np.multiply(1.0/(n+1.0),covar[n]) for n in range(d)])
	# print "covar:",covar
	x = np.random.multivariate_normal(mean0,covar,N)
	# print x.shape
	x = np.matrix(np.transpose(x))
	# print x.shape
	eps=np.matrix(np.random.normal(0,2,N))
	# print eps.shape, eps
	param=np.ones(d)
	# print param.shape, param
	y_mid= np.matrix(np.dot(param,x))
	y=np.transpose(y_mid+eps)
	# print("x:",x.shape," type:",type(x),x)
	# print("y:",y.shape," type:",type(y),y)
	# x(d,N)  y(N,1)
	return x,y

def gen_coefficients(d=48):
	v = np.random.rand(d)
	# print v
	index = v.argsort()[int(-3*d/4):]
	# print index
	index1 = v.argsort()[:int(d/4)]
	# print index1
	v1 = np.random.uniform(0.6,1,int(d/4))
	# print v1
	v2 = np.random.uniform(0,0.2,int(3*d/4))
	# print v2
	for i in range(int(3*d/4)):
		v[index[i]]=v2[i]
	for i in range(int(d/4)) :
		v[index1[i]]=v1[i]
	# print("v:",v.shape," type:",type(v),v)
	# print np.matrix(v),np.matrix(v).shape
# v(d,1)
	return np.transpose(np.matrix(v))

def data_generator2(N,v,d=48,mean=0,var=1.0):
	x=[]
	for i in range(d):
		x.append(np.random.normal(mean,var,N))
	x = np.matrix(x)
	eps=np.random.normal(0,2,N)
	# print "eps:",eps

	y=np.transpose(np.matrix(np.dot(np.transpose(v),x)+eps))

	# print "x:",x.shape," type:",type(x),x
	# print "y:",y.shape," type:",type(y),y	
	# print "v:",v.shape," type:",type(v),v
# x(d,N) y(N,1) v(d,1)
	return x, y ,v

if __name__ == '__main__':
	print("****************************************")
	x,y=data_generator1(10)
	# print("***************************************")
	# v = gen_coefficients()
	# print("****************************************")
	# x1,y1,v1 = data_generator2(10,v)
