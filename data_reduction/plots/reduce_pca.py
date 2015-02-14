import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
from scipy.cluster.vq import kmeans, kmeans2, whiten

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
#print pca
X_r = pca.fit(X).transform(X)
print X_r

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
plt.legend()
plt.xlabel('sepal width (cm)')
plt.ylabel('petal length (cm)')
plt.title('PCA of IRIS dataset')

plt.show()

#applying the kmeans method used in the knn exercise.
N = len(X_r)
w = whiten(X_r)
k = 3 
i = 40

cluster_centroids1, distortion = kmeans(w, k, iter=i)
cluster_centroids2, closest_centroids = kmeans2(w, k, iter=i)

from scipy.spatial.distance import cdist
D = cdist(X_r, cluster_centroids1, 'euclidean')
sum(np.min(D, axis=1))

plt.figure()
plt.scatter(cluster_centroids1[:, 0], cluster_centroids1[:, 1], c='r', s=100)
plt.legend()
plt.title('PCA cluster centroids')
plt.show()

plt.figure()
plt.scatter(X_r[:, 0], X_r[:, 1], c=closest_centroids, s=100)
plt.legend()
plt.title('Clustering Iris PCA ')
plt.show()

print D #cdist(coordinates, cluster_centroids1, 'euclidean')
print 'k =', k #3
print 'N =', N #150
print len(cluster_centroids1) #3
'''The PCA treatment of the iris data resulted in better segregation of the flower species as compared to the raw data.'''