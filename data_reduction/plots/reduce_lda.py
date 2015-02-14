import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA
#import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans, kmeans2, whiten


iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names


lda = LDA(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('LDA of IRIS dataset')

plt.show()

#Applying the kmeans method from earlier exercise

N = len(X_r2)
w = whiten(X_r2)
k = 3 
i = 40

cluster_centroids1, distortion = kmeans(w, k, iter=i)
cluster_centroids2, closest_centroids = kmeans2(w, k, iter=i)

from scipy.spatial.distance import cdist
D = cdist(X_r2, cluster_centroids1, 'euclidean')
sum(np.min(D, axis=1))

plt.figure()
plt.scatter(cluster_centroids1[:, 0], cluster_centroids1[:, 1], c='r', s=100)
plt.legend()
plt.title('LDA cluster centroids')
plt.show()

plt.figure()
plt.scatter(X_r2[:, 0], X_r2[:, 1], c=closest_centroids, s=100)
plt.legend()
plt.title('Clustering Iris LDA ')
plt.show()

print D #cdist(coordinates, cluster_centroids1, 'euclidean')
print 'k =', k 
print 'N =', N 
print len(cluster_centroids1) 
'''The results indicate that the the LDA reduction of the iris data prior to the kmeans treatment produced tighter/better clustering as compared to the raw data.'''