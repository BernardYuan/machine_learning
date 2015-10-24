#Yuan Bowei, 1001916, 22 Sept 2015
import numpy
import matplotlib.pyplot as plt
from scipy import spatial

#this function is for generating data points and their labels
def generate_data(n=5000):   
	x = []
	y = []
	labels = []
	for i in list(range(n)):
		a = numpy.random.uniform(-1,1,2) #get the data points in uniform distribution
		b = numpy.random.random_integers(0,99)   #get a random integer for possibility
		if a[1]> 0:    # generating the label
			if b>24:
				label = 1
			else :
				label = -1
		elif a[1]<0:
			if b>24:
				label=-1
			else :
				label=1
		else:
			if b>49:
				label=1
			else :
				label = -1
		x.append(a[0]);
		y.append(a[1]);
		labels.append(label)
	return x,y,labels

#This part of code is to generate training data and labels
x_train,y_train,colour_train=generate_data()

#to plot the training data
plt.scatter(x_train,y_train,s=30,c=colour_train)
plt.title("Training Data Plot")
plt.show()

#generate testing data and labels
x_test,y_test,colour_test=generate_data(1000)
#plot the test data and labels
plt.scatter(x_test,y_test,s=30,c=colour_test)
plt.title("Test Data Plot")
plt.show()

# The kNN function
def kNN(xTrain, yTrain, xTest, k=100):
	yTest=[]
	tree = spatial.KDTree(list(zip(xTrain[0],xTrain[1])))
	points = numpy.transpose([xTest[0],xTest[1]])
	for i in range(len(points)):
		distance,index=tree.query(points[i],k)
		weight=0
		for j in range(k):
			weight+=yTrain[index[j]]/distance[j]
		if weight>0:
			yTest.append(1)
		elif weight<0:
			yTest.append(-1)
		else :
			if numpy.random.random_integers()%2==1:
				yTest.append(1)
			else :
				yTest.append(-1)
	return yTest

# a function for calculating the score (correction rate):
def score(target, predict):
	sco=0
	for i in range(len(target)):
		if target[i]==predict[i]:
			sco+=1
	return sco/len(target)

# predict the labels
prediction1 = kNN([x_train,y_train],colour_train,[x_test,y_test])

#calculate the score1
print('Score 1(original data):',score(colour_test,prediction1))

#plot the prediction
plt.scatter(x_test,y_test,s=30,c=prediction1)
plt.title("Prediction1 Plot (original training data and test data)")
plt.show()

#scaled x-axis of training data
x_train_scaled = numpy.multiply(1000,x_train)
x_test_scaled = numpy.multiply(1000,x_test)

#prediction after scaling
prediction2 = kNN([x_train_scaled,y_train],colour_train,[x_test_scaled,y_test])
print('Score 2(after scaling x-axis by 1000):',score(colour_test,prediction2))

#plot the new prediction
plt.scatter(numpy.multiply(1000,x_test),y_test,s=30,c=prediction2)
plt.title('Prediction2 Plot (x_axis scaled by 1000)')
plt.show()

#compute the standard deviations of both dimensions
x_train_deviation=numpy.std(x_train)
y_train_deviation=numpy.std(y_train)
print("Standard deviation of both dimensions (training data):",(x_train_deviation,y_train_deviation))
#training data after dividing standard deviation
x_train_div=numpy.multiply(1/x_train_deviation,x_train)
y_train_div=numpy.multiply(1/y_train_deviation,y_train)
print("Standard deviation of both dimensions (training data) after dividing standard deviation:",(numpy.std(x_train_div),numpy.std(y_train_div)))

#the standard deviation of test data
x_test_deviation=numpy.std(x_test)
y_test_deviation=numpy.std(y_test)

#test data after dividing standard deviation
x_test_div=numpy.multiply(1/x_test_deviation,x_test)
y_test_div=numpy.multiply(1/y_test_deviation,y_test)
print("Standard deviation of both dimensions (test data) after dividing standard deviation:",(numpy.std(x_test_div),numpy.std(y_test_div)))
#run prediction after dividing the standard deviation
prediction3=kNN([x_train_div,y_train_div],colour_train,[x_test_div,y_test_div])

print("Score3 (after dividing the standard deviations):",score(colour_test,prediction3))
plt.scatter(x_test_div,y_test_div,s=30,c=prediction3)
plt.title("Prediction3 Plot (after the less general normalization)")
plt.show()

#apply this normalization method to the x-axis scaled dataset
print("std of scaled x-axis of training data:",numpy.std(x_train_scaled))
print("std of scaled x-axis of testing data:",numpy.std(x_test_scaled))

x_train_scaled_div=numpy.multiply(1/numpy.std(x_train_scaled),x_train_scaled)
x_test_scaled_div = numpy.multiply(1/numpy.std(x_test_scaled),x_test_scaled)
print("std of scaled&&normalized training data:",(numpy.std(x_train_scaled_div),numpy.std(y_train_div)))
print("std of scaled&&normalized testing data:",(numpy.std(x_test_scaled_div),numpy.std(y_test_div)))

prediction4=kNN([x_train_scaled_div,y_train_div],colour_train,[x_test_scaled_div,y_test_div])
print("score after the scaled data dividing std:",score(colour_test,prediction4))


#apply the more general method to original dataset
x_train_mean=numpy.mean(x_train)
y_train_mean=numpy.mean(y_train)
x_test_mean=numpy.mean(x_test)
y_test_mean=numpy.mean(y_test)

x_train_gennorm=numpy.multiply(1/x_train_deviation,[x-x_train_mean for x in x_train])
y_train_gennorm=numpy.multiply(1/y_train_deviation,[y-y_train_mean for y in y_train])
x_test_gennorm=numpy.multiply(1/x_test_deviation,[x-x_test_mean for x in x_test])
y_test_gennorm=numpy.multiply(1/y_test_deviation,[y-y_test_mean for y in y_test])

#std after the more general normalization
print("std of training data after more general normalization:",(numpy.std(x_train_gennorm),numpy.std(y_train_gennorm)))
print("std of testing data after more general normalization:",(numpy.std(x_test_gennorm),numpy.std(y_test_gennorm)))
#prediction after the more general normalization:
prediction5=kNN([x_train_gennorm,y_train_gennorm],colour_train,[x_test_gennorm,y_test_gennorm])
print("score after the more general normalization:",score(colour_test,prediction5))

plt.scatter(x_test_gennorm,y_test_gennorm,s=30,c=prediction5)
plt.title("Prediction5 Plot (after the more general normalization)")
plt.show()

#the inspired euclidean measure, which should shrink the x-axis by the times it was scaled
def euclidean_measure(point1, point2):
	return sqrt(((point1[0]-point2[0])/1000)**2+(point[1]-point2[1])**2)
