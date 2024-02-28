import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#read from file and split into three 1d arrays
mydata = np.genfromtxt('inputdata.csv', delimiter=' ', skip_header=1)
aon=mydata[:,0] #1st column, online ads
atv=mydata[:,1] #2nd column, tv ads
sal=mydata[:,2] #3rd column, sales

#shuffle the arrays
indx=np.arange(aon.size)
np.random.shuffle(indx)
saon=aon[indx]
satv=atv[indx]
ssal=sal[indx]

#split into training and testing data
taon=saon[0:90] #training data
tatv=satv[0:90]
tsal=ssal[0:90]

caon=saon[91:] #test data
catv=satv[91:]
csal=ssal[91:]

#uniform arrays for plotting in respect of x and y
xaxi=np.arange(11)
yaxi=np.arange(11)

#Model 1 Sales v Online advertising

#get the fit
x = taon.reshape((-1, 1))
y=tsal
lrmodel = LinearRegression().fit(x, y)

#get a and b
lrminter=lrmodel.intercept_
lrmcoefs=lrmodel.coef_

#plot data and fit
plt.figure(1)
plt.scatter(taon,tsal)
plt.scatter(caon,csal, c='red')
plt.plot(xaxi, lrminter+lrmcoefs*xaxi, color='black')
plt.title('Sales revenue v advertising', fontsize=25)
plt.ylabel('Sales', fontsize=15)
plt.xlabel('Online advertising', fontsize=15)

#predict some values and use testing data to calculate the MSE
y_pred=lrminter+lrmcoefs*caon
print("Mean squared error: %.2f" % mean_squared_error(csal, y_pred))

#Model 2 Sales v TV advertising, same as in model 1

x = tatv.reshape((-1, 1))
y=tsal
lrmodel = LinearRegression().fit(x, y)

lrminter=lrmodel.intercept_
lrmcoefs=lrmodel.coef_

plt.figure(2)
plt.scatter(tatv,tsal)
plt.scatter(catv,csal, c='red')
plt.plot(xaxi, lrminter+lrmcoefs*xaxi, color='black')
plt.title('Sales revenue v advertising', fontsize=25)
plt.ylabel('Sales', fontsize=15)
plt.xlabel('TV advertising', fontsize=15)

#predict some values and use testing data to calculate the MSE
y_pred=lrminter+lrmcoefs*catv
print("Mean squared error: %.2f" % mean_squared_error(csal, y_pred))

#Model 3 Sales v TV advertising

#x now contains two independent (feature) variables
x = np.vstack((taon,tatv)).T
y=tsal
#fit the data
lrmodel = LinearRegression().fit(x, y)

#extract a, b, c
lrminter=lrmodel.intercept_
lrmcoefs=lrmodel.coef_

#plot data and fit
fig3=plt.figure(3)
ax = fig3.add_subplot(projection='3d')
ax.scatter(taon,tatv,tsal)
ax.scatter(caon,catv,csal, color='red')
plt.xlabel('Online advertising', fontsize=15)
plt.ylabel('TV advertising', fontsize=15)
plt.title('Sales', fontsize=15)
#create mesh for surface plot
xx, yy = np.meshgrid(xaxi, yaxi)
zz=lrminter+lrmcoefs[0]*xx+lrmcoefs[1]*yy
ax.plot_surface(xx, yy, zz, color='grey')

#predict some values and use testing data to calculate the MSE
y_pred=lrminter+lrmcoefs[0]*caon+lrmcoefs[1]*catv
print("Mean squared error: %.2f" % mean_squared_error(csal, y_pred))





