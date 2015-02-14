import numpy as np
from scipy.cluster.vq import kmeans,vq
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

#reference for some code of the code used here http://stackoverflow.com/questions/6645895/calculating-the-percentage-of-variance-measure-for-k-means
data=np.genfromtxt("un_2.csv", delimiter = ',')
data=np.ma.masked_array(data, np.isnan(data))#masks nans for math
print np.mean(data)#1616.198

# Clusters data into K=1-10 clusters 
#K, KM, centroids,D_k,cIdx,dist,avgWithinSS = kmeans.run_kmeans(X,10)

K = range(1,10)

 # scipy.cluster.vq.kmeans
KM = [kmeans(data,k) for k in K] # apply kmeans 1 to 10
centroids = [cent for (cent,var) in KM]   # cluster centroids

D_k = [cdist(data, cent, 'euclidean') for cent in centroids]

cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/data.shape[0] for d in dist]  

print avgWithinSS#Prints the distances for 10 closest neighbors.

kIdx = 2

#Visualization the results.
# cool elbow curve  
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')
ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
      markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
tt = plt.title('Elbow for K-Means clustering')  
plt.show()

# scatter plot
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(data[:,2],data[:,1], s=30, c=cIdx[k])
clr = ['m','y','k']
#clr = ['b','g','r','c','m','y','k']
for i in range(K[kIdx]):
    ind = (cIdx[kIdx]==i)
    ax.scatter(data[ind,0],data[ind,3], s=30, c=clr[i], label='Cluster %d'%i)
plt.xlabel('GDPperCapita')
plt.ylabel('lifeMale, lifeFemale, infantMortality')
plt.title('UN dataset, KMeans clustering with K=%d' % K[kIdx])
plt.legend()
plt.show()
