import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, kmeans2, whiten
from scipy.spatial.distance import cdist

#Reading in the csv file via pandas as a dataframe.
df = pd.read_csv('un.csv')
df.dropna(inplace=True)
df.head()

#Reading the dataframe as a matrix for plotting and further manipulation.
coordinates = df.as_matrix(columns=['infantMortality', 'lifeMale', 'lifeFemale','GDPperCapita'])
plt.figure()
plt.scatter(coordinates[:,3], coordinates[:,0], c='c', s=100)
plt.xlabel('GDPperCapita')
plt.ylabel('infantMortality')
plt.show()

plt.figure()
plt.scatter(coordinates[:,3], coordinates[:,1], c='c', s=100)
plt.xlabel('GDPperCapita')
plt.ylabel('lifeMale')
plt.show()

plt.figure()
plt.scatter(coordinates[:,3], coordinates[:,2], c='c', s=100)
plt.xlabel('GDPperCapita')
plt.ylabel('lifeFemale')
plt.show()

#Data preparation and terms for calculating closest centroids.
N = len(coordinates)
w = whiten(coordinates)
k = 3 
i = 20

cluster_centroids1, distortion = kmeans(w, k, iter=i)
cluster_centroids2, closest_centroids = kmeans2(w, k, iter=i)

#Visualizing the cluster and closest centroids as scatter plots.
plt.figure()
plt.scatter(cluster_centroids1[:,3], cluster_centroids1[:,0], c='r', s=100)
plt.xlabel('GDPperCapita')
plt.ylabel('infantMortality')
tt = plt.title('Cluster Centroids') 
plt.show()

plt.figure()
plt.scatter(cluster_centroids1[:,3], cluster_centroids1[:,1], c='r', s=100)
plt.xlabel('GDPperCapita')
plt.ylabel('lifeMale')
tt = plt.title('Cluster Centroids')
plt.show()

plt.figure()
plt.scatter(cluster_centroids1[:,3], cluster_centroids1[:,2], c='r', s=100)
plt.ylabel('GDPperCapita')
plt.xlabel('lifeFemale')
tt = plt.title('Cluster Centroids')
plt.show()

plt.figure()
plt.scatter(coordinates[:,3], coordinates[:,0], c=closest_centroids, s=100)
plt.xlabel('GDPperCapita')
plt.ylabel('infantMortality')
tt = plt.title('Closest Centroids') 
plt.show()

plt.figure()
plt.scatter(coordinates[:,3], coordinates[:,1], c=closest_centroids, s=100)
plt.xlabel('GDPperCapita')
plt.ylabel('lifeMale')
tt = plt.title('Closest Centroids')
plt.show()

plt.figure()
plt.scatter(coordinates[:,3], coordinates[:,2], c=closest_centroids, s=100)
plt.xlabel('GDPperCapita')
plt.ylabel('lifeFemale')
tt = plt.title('Closest Centroids')
plt.show()

#Function to determine seuclidean distance
from scipy.spatial.distance import cdist
D = cdist(coordinates, cluster_centroids1, 'euclidean')
sum(np.min(D, axis=1))


print D #cdist(coordinates, cluster_centroids1, 'euclidean')
print 'k =', k #3
print len(cluster_centroids1) #3
print len(closest_centroids)# 39
