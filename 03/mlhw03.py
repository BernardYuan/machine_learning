# Homework 3
import numpy as np
import scipy as scp
from scipy import spatial
import matplotlib.pyplot as plt
import random
import math
import sys
# generate one garussian
# input 2 numbers
# output one number
def gengaussian(mean, variance):
	x1 = -1
	std = math.sqrt(variance)
	while x1<=0:
		x1 = np.random.normal(mean,std)
	return x1
#params: numdata int, probs(1,3), means(2,3), sigmadiags(2,3)
#returns: x: matrix(2,numdata) | cluster: array(numdata,)
def gensomemixturedata(numdata, probs, means, sigmadiags):
	x = [[],[]]
	cluster = []
	p1=probs[0]
	p2=p1+probs[1]
	p3=p2+probs[2]

	for i in range(numdata):
		r = random.random()
		if(r>=0 and r<p1):
			x[0].append(gengaussian(means[0,0],sigmadiags[0,0]))
			x[1].append(gengaussian(means[1,0],sigmadiags[1,0]))
			cluster.append(0)
		elif (r>=p1 and r<p2):
			x[0].append(gengaussian(means[0,1],sigmadiags[0,1]))
			x[1].append(gengaussian(means[1,1],sigmadiags[1,1]))
			cluster.append(1)
		elif (r>=p2 and r<p3):
			x[0].append(gengaussian(means[0,2],sigmadiags[0,2]))
			x[1].append(gengaussian(means[1,2],sigmadiags[1,2]))
			cluster.append(2)

	return np.matrix(x),np.array(cluster)
# objective function
# input: x: matrix(2,numdata) | cluster: array(numdata) | center: matrix(2,k)
# output: res: int 
def objective(x,cluster,center,dist):
	res = []
	for i in range(x.shape[1]):
		c = cluster[i]
		res.append(dist(x[:,i],center[:,c]))
	return np.sum(res)

# distance measure
# input: p1: matrix(2,1)|p2: matrix(2,1)
# output: dist double
def dist0(p1, p2):
	p1 = np.ravel(p1)
	p2 = np.ravel(p2)
	s = 0
	for i in range(p1.shape[0]):
		s+= (p1[i]-p2[i])**2
	return s
# chi square distance
def dist1(p1, p2):
	p1 = np.ravel(p1)
	p2 = np.ravel(p2)
	n1 = 0
	n2 = 0
	s = 0
	for i in range(p1.shape[0]):
		n1 = (p1[i]-p2[i])**2.0
		n2 = p1[i]+p2[i]
		if(n1==0 and n2==0):
			continue
		else:
			s+=n1/n2
	return s

# update centers:
# input: x: matrix(2,numdata)| clusters: array(numdata) | k int
# output: center: matrix(2,k)

# update to any point by calculating the physics center
def updatecenters0(x,clusters,k):
	center = []
	for i in range(k):
		index = clusters==i
		index = np.array(range(x.shape[1]))[index]
		if(i==0):
			center = np.average(x[:,index],axis=1)
		else:
			center=np.concatenate((center,np.average(x[:,index],axis=1)),axis=1)
	return np.matrix(center)

# update to a point existing in datapoints with the given distance function
def updatecenters1(x,clusters,k,dist):
	center = []
	for i in range(k):
		index = clusters==i
		index = np.array(range(x.shape[1]))[index]
		subgroup = x[:,index]
		# calculate the distance matrix
		dm = spatial.distance.cdist(subgroup.T,subgroup.T,dist)
		total = np.sum(dm,axis=1) # (1,numdata)
		no = np.argmin(total)
		if(i==0):
			center = subgroup[:,no]
		else:
			center = np.concatenate((center,subgroup[:,no]),axis=1)
	return center

# initializer functions:
# initialize randomly
# input: x matrix(2,numdata) | k int
# output: centers matrix(2,k)
# NOTE: theoreticall in kmeans algorithm the initail points can be drawn absolutely arbitarily,
# But in the real yimplementation, drawing them randomly from existing datapoints makes the most sense
def randinit(x,k):
	population = range(x.shape[1])
	samp = random.sample(population,k)
	centers = x[:,samp[0]]
	for i in samp[1:]:
		centers=np.concatenate((centers,x[:,i]),axis=1)
	return np.matrix(centers)

# kmean++ initializer
# calculate the probability
# input x matrix(2,numdata), centers matrix(2,0<=n<=k)
# output probs array(numdata)
def calprob(x,centers):
	prob = np.zeros(x.shape[1])
	for i in range(centers.shape[1]):
		dist = np.zeros(x.shape[1])
		for j in range(x.shape[1]):
			dist[j]=math.sqrt(dist0(x[:,j],centers[:,i]))
		dist = dist * 1.0 / np.sum(dist)
		prob+=dist
	prob = prob * 1.0 / centers.shape[1]
	prob = np.cumsum(prob)
	return prob
# input x matrix(2,numdata) | k int
# output centers matrix(2,k)
def kmeanpp(x,k):
	# initially, choose one random point, to initialize the set
	centers = randinit(x,1)
	# apply k mean algorithm
	for i in range(1,k):
		prob = calprob(x,centers)
		r = random.random()
		for j,l in enumerate(prob):
			if r < l:
				centers=np.concatenate((centers,x[:,j]),axis=1)
				break
	return centers

#K-means clustering
# input: dataset: matrix(2,numdata) | k: int | initial: initializer function
# output: group: array(numdata) | centers: matrix(2,k)| obj: double
def kmeans(dataset, k, initial,dist):
	group = np.zeros(dataset.shape[1])
	centers = initial(dataset,k)
# optimize start
	delta = sys.maxint
	obj_last = 0
	itera = 0
	while(delta>1e-4):
		# print "iteration:", itera
		itera+=1
		for i in range(dataset.shape[1]):
			mindist = sys.maxint;
			mindistno = 0
			for j in range(k):
				tempdist = dist(dataset[:,i],centers[:,j])
				if tempdist<=mindist:
					mindist = tempdist
					mindistno = j
			group[i]=mindistno
		obj_new = objective(dataset,group,centers,dist)
		# print "obj:",obj_new
		delta = abs(obj_new-obj_last)
		# print "delta:",delta
		obj_last=obj_new
		centers=updatecenters0(dataset,group,k)
		# print "centers:", centers
	return group,centers,obj_last

# pairwise distance:
# input c matrix(2,k), k = number of centers
# output pairwise distances 
def pairwisdist(c):
	s = 0
	k = c.shape[1]
	for i in range(k):
		for j in range(i):
			s += dist0(c[:,i],c[:,j])
	return s*2.0/(k*(k-1.0))

# histogram data
# input x(2,numdata)
# output (3,numdata)
# map the data to a three dimension, but the third dimension is not a degree of freedom
def histogramdata(x):
	x = np.array(x)
	mx1 = np.amax(x[0])
	mx2 = np.amax(x[1])
	r1 = x[0]/(mx1+0.01)
	r2 = x[1]/(mx2+0.01)
	r3 = np.ones(x.shape[1])-0.5*r1 -0.5*r2
	return np.matrix([r1,r2,r3])

# kmedoids algorithm
# input: dataset(dimension 2/3,numdata), k:int, initial:initializer function, dist:distance function
# output: group(numdata,), centers (dimension 2/3, k), obj_last double
def kmedoids(dataset,k,initial,dist):
	group = np.zeros(dataset.shape[1])
	centers = initial
# optimize start
	delta = sys.maxint
	obj_last = 0
	itera = 0
	while(delta>0):     #till the delta stops changing
		print "iteration:", itera
		itera+=1
		for i in range(dataset.shape[1]):
			mindist = sys.maxint;
			mindistno = 0
			for j in range(k):
				tempdist = dist(dataset[:,i],centers[:,j])
				if tempdist<=mindist:
					mindist = tempdist
					mindistno = j
			group[i]=mindistno
		obj_new = objective(dataset,group,centers,dist)
		print "obj:",obj_new
		delta = abs(obj_new-obj_last)
		print "delta:",delta
		obj_last=obj_new
		centers=updatecenters1(dataset,group,k,dist)
		# print "centers:", centers
	return group,centers,obj_last


if __name__ == '__main__':

#2 b)
# data generator:
	# d,c = gensomemixturedata(2000,np.array([0.15,0.3,1-0.45]),np.array([[3,6,5.1],[3,3.6,9]]),np.array([[1,1,1],[1,0.5,1.5]]))
	# plt.title("Original cluster")
	# plt.scatter(d[0,:],d[1,:],c=c)
	# plt.show()
# save data
	# np.save('points.npy',d)
	# np.save('clusters.npy',c)

# load data, pls uncomment this to get the testing data
	d = np.load('points.npy')		#datapoints
	c = np.load('clusters.npy')	#the cluster of the datapoints of the same indices

#2 c)
# plotting result showing points and centers, with selecting initial points randomly
	# d = np.matrix(d)
	# for i in [2,3,4,8,16]:
	# 	r1,r2,_= kmeans(d, i, randinit,dist0)
	# 	plt.figure(i)
	# 	plt.title("clustering with "+str(i)+" centers")
	# 	plt.scatter(d[0,:],d[1,:],c=r1)
	# 	plt.plot(r2[0,:],r2[1,:],'mp')
	# 	plt.show()
# plotting the result with kmeanpp initializer:
	# d = np.matrix(d)
	# for i in [2,3,4,8,16]:
	# 	r1,r2,_= kmeans(d, i, kmeanpp,dist0)

	# 	plt.figure(i)
	# 	plt.title("clustering with "+str(i)+" centers")
	# 	plt.scatter(d[0,:],d[1,:],c=r1)
	# 	plt.plot(r2[0,:],r2[1,:],'mp')
	# 	plt.show()
# 2 d) e) f)
#calculating and plotting the averaged pairwise distance and objective of km++ algorithm
#	d = np.matrix(d)
#	apwdist = []
#	aobj = []
#	for i in [2,3,4,8,16]:
#		print "number of centers:",i
#		pwdist = []
#		obj = []
#		for j in range(10):
#			print "iteration:",j
#			g,c,o=kmeans(d,i,kmeanpp,dist0)
#			print "centers:",c
#			print pairwisdist(c)
#			pwdist.append(pairwisdist(c))
#			obj.append(o)
#		print "i:",i
#		print "pairwise distance:",pwdist
#		print "average pairwise distance:",np.average(pwdist)
#		print "variance:",np.var(pwdist)
#		print "objective:",obj
#		print "average objective:",np.average(obj)
#		print "variance:",np.var(obj)
#		apwdist.append(np.average(pwdist))
#		aobj.append(np.average(obj))
#	print "apwdist:",apwdist
#	print "aobj:",aobj
#	plt.figure(1)
#	plt.plot([2,3,4,8,16],apwdist,'ro-')
#	plt.show()
#	plt.figure(2)
#	plt.plot([2,3,4,8,16],aobj,'ro-')
#	plt.show()

# 3 b)
	# mx1 = np.amax(d[0])
	# mx2 = np.amax(d[1])
# 3 c) d) e)
# map to 3 dimension
	# d=histogramdata(d)
# randomly initialize points and save in the list

	# init = []
	# for i in [2,3,4,8,16]:
	# 	init.append(randinit(d,i))
# run k-medoids with the initial points, and use squared distances
	# 	g,c,o = kmedoids(d,i.shape[1],i,dist0)
	# 	plt.figure(i.shape[1])
	# 	plt.scatter(d[0,:],d[1,:],c=g)
	# 	plt.plot(c[0,:],c[1,:],'mp')
	# 	plt.show()
# run k-medoids with the initial points, and use chi quared distances
	# 	g1, c1, o1 = kmedoids(d,i.shape[1],i,dist1)
	# 	plt.figure(100+i.shape[1])
	# 	plt.scatter(d[0,:],d[1,:],c=g1)
	# 	plt.plot(c1[0,:],c1[1,:],'mp')
	# 	plt.show()
