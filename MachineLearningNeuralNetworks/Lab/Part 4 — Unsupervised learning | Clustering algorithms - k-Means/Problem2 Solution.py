
from matplotlib import image as img
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from sklearn.cluster import KMeans

# load the MxN image into MxNx3 float array
imgdata = np.asarray(img.imread('sunspot.jpg'))
imgdata=imgdata.astype(float)

#slice the MxNx3 float array into three MxN arrays for each colour channel
rr=imgdata[:,:,0]
gg=imgdata[:,:,1]
bb=imgdata[:,:,2]

#create 'Intensity map' based on colour maps
gs=np.sqrt((rr*rr+gg*gg+bb*bb)/3.0)

#create X and Y arrays
x=np.arange(800.0)*10.0
y=np.arange(600.0)*10.0

#plot greyscale image
plt.figure(0)
plt.contourf(x,y,gs, cmap='Greys_r', levels=128)

#Create X,Y mesh grid
xx, yy = np.meshgrid(x,y)

#Smooth the colour maps and GS map using 2D convolution
wind=51
kern=np.full((wind,wind),1.0)
kern=kern/np.sum(kern)
rr=convolve2d(rr,kern,mode='same', boundary='fill', fillvalue=255)
gg=convolve2d(gg,kern,mode='same', boundary='fill', fillvalue=255)
bb=convolve2d(bb,kern,mode='same', boundary='fill', fillvalue=255)
gs=convolve2d(gs,kern,mode='same', boundary='fill', fillvalue=255)

#plot smoothed greyscale image
plt.figure(1)
plt.contourf(x,y,gs, cmap='Greys_r', levels=128)

#Create imput data for kMeans
#gs.ravel() converts 2D MxN array into 1D array with MxN elements
#[] brackets make gs 'formally' 2D (kMeans wants at least two dimensions in the data)
inputdata=np.transpose([gs.ravel()])

#Create kmeans model with two clusters and label pixels
km = KMeans(n_clusters=2)
lbls = km.fit_predict(inputdata)

#converts 1D array with MxN elements into 2D MxN array
lblmap=lbls.reshape((600,800))

#plot the label map
plt.figure(2)
plt.contourf(x,y,lblmap, cmap='cividis', levels=128)

#Calculate the number of pixels with label=1 
#!!!!!!!BUT CHECK WHICH LABEL IS FOR SUNSPOT!!!!!!
Npixels=np.sum(lblmap==1)

#Area of each pixel is 0.0001 Sq.Mm., hence, the area is Npixels*0.0001
print('Sunspot area, sq Mm:', Npixels*0.0001)