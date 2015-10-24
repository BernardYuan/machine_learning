#Yuan Bowei, 1001916, 22 Sept,2015
import numpy as np
import scipy as scp 
import matplotlib.pyplot as plt 
#generate the 2D gaussian
row1=np.random.normal(1,1,100)
row2=np.random.normal(3,1,100)
X1=[row1,row2]
plt.plot(X1[0],X1[1],'bs')
plt.title("Original Dataset")
plt.show()

#Linear Mapping A1, mirrors along the y-axis
A1=[[-1,0],[0,1]]
X2=np.dot(A1,X1)
plt.plot(X1[0],X1[1],'bs')
plt.plot(X2[0],X2[1],'rs')
plt.title("Mirrors along the y-axis")
plt.show()

#Linear Mapping A2, scale x-axis by 0.5
A2=[[0.5,0],[0,1]]
X3=np.dot(A2,X1)
plt.plot(X1[0],X1[1],'bs')
plt.plot(X3[0],X3[1],'rs')
plt.title("scale x-axis by 0.5")
plt.show()

#Linear Mapping A3, rotate data by 45 degree clockwise
A3 = [[np.cos(np.pi/4),np.sin(np.pi/4)],[-np.sin(np.pi/4),np.cos(np.pi/4)]]
X4 = np.dot(A3,X1)

plt.plot(X1[0],X1[1],'bs')
plt.plot(X4[0],X4[1],'rs')
plt.title("rotate by 45 degree clockwise")
plt.show()

#Linear Mapping A4, mirrors along x-axis
A4=[[1,0],[0,-1]]
X5 = np.dot(A4,X1)
plt.plot(X1[0],X1[1],'bs')
plt.plot(X5[0],X5[1],'rs')
plt.title("Mirror along x-axis")
plt.show()

# A5, a combination
A5 = np.dot(np.dot(A2,A1),A4)
X6 = np.dot(A5,X1)
plt.plot(X1[0],X1[1],'bs')
plt.plot(X6[0],X6[1],'rs')
plt.title("After transformed by A5")
plt.show()
