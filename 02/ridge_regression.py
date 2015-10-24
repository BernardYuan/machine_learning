import numpy as np
import scipy as scp
import matplotlib.pyplot as plt 
import math
import gendata as gd
import random


#this function implements w= (XTX+lambdaI)^-1 * XT * y
#But attention in this function X(N,d),y(N,1), so x need to be transposed
#return value w(d,1)
#Input x(d,N) y(N,1)
def train_ridgeregression(x,y,l):
	x=np.transpose(x)
	identity = np.identity(x.shape[1])
	xtx=np.matrix(np.dot(np.transpose(x),x))
	xtx=xtx+np.multiply(l,identity)
	xtxinv = np.linalg.inv(xtx)
	w=np.dot(xtxinv,np.transpose(x))
	w=np.dot(w,y)
# w(d,1)
	return w


# This funciton calculates the MSE
# x(d,N) y(N,1) w(d,1)
def cal_mse(test_x,test_y,w):
	me = test_y-np.transpose(np.dot(np.transpose(w),test_x))
	m1 = np.dot(np.transpose(me),me)
	return m1[0][0]/test_x.shape[1]

# This function performs ridge regression with lambda=1e-30 and 5 respectively
# and calculates average MSE
def RidgeRegressionData1(run=10,train=100,test=1000):
	mse=[np.zeros(run),np.zeros(run),np.zeros(run),np.zeros(run),np.zeros(run),np.zeros(run)]

	for i in range(run) :
		train_x,train_y=gd.data_generator1(train)
		test_x,test_y=gd.data_generator1(test)
		la=1e-30
		w1=train_ridgeregression(train_x,train_y,la)
		#w(d,1)   y(N,1)
		# me = test_y-np.transpose(np.dot(np.transpose(w1),test_x))
		# m1 = np.dot(np.transpose(me),me)
		mse[0][i]+=cal_mse(test_x,test_y,w1)

		la = 5
		w2 = train_ridgeregression(train_x,train_y,la)
		# me = test_y-np.transpose(np.dot(np.transpose(w2),test_x))
		# m2 = np.dot(np.transpose(me),me)
		mse[1][i]+=cal_mse(test_x,test_y,w2)
# jusr for testing, the convergence of different lambdas 
		la = 20
		w3 = train_ridgeregression(train_x,train_y,la)
		mse[2][i] += cal_mse(test_x,test_y,w3)

		la = 100
		w4 = train_ridgeregression(train_x,train_y,la)
		mse[3][i] += cal_mse(test_x,test_y,w4)

		la = 1000
		w5 = train_ridgeregression(train_x,train_y,la)
		mse[4][i] += cal_mse(test_x,test_y,w5)

		la = 10000
		w6 = train_ridgeregression(train_x,train_y,la)
		mse[5][i] += cal_mse(test_x,test_y,w6)

	# print("mse:",np.matrix(mse))
	avg0 = np.average(mse[0])
	avg1 = np.average(mse[1])
	avg2 = np.average(mse[2])
	avg3 = np.average(mse[3])
	avg4 = np.average(mse[4])
	avg5 = np.average(mse[5])
	# print("avg0:",avg0)
	# print("avg1:",avg1)
	return avg0, avg1, avg2, avg3, avg4, avg5


# This function performs holdout method
def holdout(train=400,test=100,la=1e-2):

	coe=gd.gen_coefficients()
	data_x,data_y,_=gd.data_generator2(train+test,coe)

	train_seq = random.sample([x for x in range(train+test)],train)
	test_seq = [x for x in range(train+test) if x not in train_seq]
	
	#training data
	train_x=[data_x[:,j] for j in train_seq]
	train_x = np.transpose(np.matrix(np.ravel(train_x).reshape(train,48)))
	train_y=np.transpose(np.matrix([data_y[j,0] for j in train_seq]))
	#testing data
	test_x=[data_x[:,j] for j in test_seq]
	test_x= np.transpose(np.matrix(np.ravel(test_x).reshape(test,48)))
	test_y=np.transpose(np.matrix([data_y[j,0] for j in test_seq]))
	
	w = train_ridgeregression(train_x,train_y,la)
		# print("w:",w.shape,"data:",w)
	return cal_mse(test_x,test_y,w)

def testHoldout(run = 10):
	mse = np.zeros(run)

	for i in range(run):
		mse[i] += holdout()
	return np.average(mse),np.var(mse)

#  Cross Validation method
def crossValidation(train=400,test=100,la=1e-2,run=5):
	mse = np.zeros(run)
	coe=gd.gen_coefficients()
	data_x,data_y,_=gd.data_generator2(train+test,coe)
	seq = np.arange(train+test)
	seq = np.random.permutation(seq)
	seq = seq.reshape(int((train+test)/test),test)

	for i in range(run):
		test_x=[data_x[:,j] for j in seq[i]]
		test_x= np.transpose(np.matrix(np.ravel(test_x).reshape(test,48)))
		test_y= np.transpose(np.matrix([data_y[j,0] for j in seq[i]]))
		# print("test_x:",test_x.shape,"type:",type(test_x))
		# print("test_y:",test_y.shape,"type:",type(test_y))

		train_x = [data_x[:,j] for j in np.ravel(seq) if j not in seq[i]]
		train_x = np.transpose(np.matrix(np.ravel(train_x).reshape(train,48)))
		train_y = np.transpose(np.matrix([data_y[j,0] for j in np.ravel(seq) if j not in seq[i]]))
		# print('train_x:',train_x.shape,"type:",type(train_x))
		# print('train_y:',train_y.shape,"type:",type(train_y))

		w=train_ridgeregression(train_x,train_y,la)
		
		mse[i]+=cal_mse(test_x,test_y,w)
	return np.average(mse)

def testCrossValidation(fold=10):
	mse = np.zeros(fold)
	for i in range(fold):
		mse[i]+=crossValidation()
	return np.average(mse),np.var(mse)
# Compare CrossValidation method and holdout method
def cvh():
	A1,V1 = testHoldout()
	A2,V2 = testCrossValidation()
	print "Holdout MSE:",A1,",Variance:",V1
	print "Cross Validation: MSE:",A2,"Variance",V2

if __name__ == '__main__':
#uncomment the following code to compare the MSE of lambda=1e-30 and lambda=5
	# avg00, avg01,_,_,_,_ = RidgeRegressionData1(10,100,1000)
	# avg10, avg11,_,_,_,_ = RidgeRegressionData1(10,500,1000)
	# print "lambda=1e-30, training=100, result:",avg00
	# print "lambda=1e-30, training=500, result:",avg10
	# print "lambda=5, training=100, result:",avg01
	# print "lambda=5, training=500, result:",avg11


# uncomment and execute the following code to draw how it converge as training data grows
	# x = []
	# mse1 = np.zeros(110)
	# mse2 = np.zeros(110)
	# mse3 = np.zeros(110)
	# mse4 = np.zeros(110)
	# mse5 = np.zeros(110)
	# mse6 = np.zeros(110)

	# for i in range(110):
	# 	x.append(100+i*90)
	# 	mse1[i],mse2[i],mse3[i],mse4[i],mse5[i], mse6[i] = RidgeRegressionData1(10,100+i*90,10000)
	# plt.figure(2)
	# plt.plot(x,mse1,"r")
	# plt.plot(x,mse2,"g")
	# plt.xlabel("size of training data")
	# plt.ylabel("average MSE")
	# plt.plot(x,mse3,'y')
	# plt.plot(x,mse4,'b')
	# plt.plot(x,mse5,'m')
	# plt.plot(x,mse6,'o')
	# plt.show()

# uncomment the following code to get the comparison between holdout and crossvalidation
	cvh()