import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

df1=pd.read_csv('ofmale2.csv', names=['mlocom', 'mthigmodperc', 'mthigmotperc', 'mlean', 'mrear', 'mgroomfr', 'mgroomdur', 'mgroombout', 'mjump', 'mdefec'], header=0) 
males = np.array(df1.as_matrix(columns = None), dtype=bool).astype(np.int)

X = males
pca = PCA(n_components=2)
pca.fit(X)
PCA(copy=True, n_components=2, whiten=False)
#print(pca.explained_variance_ratio_) 
#print(pca.explained_variance_)

print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

df2=pd.read_csv('offemale2.csv', names=['flocom', 'fthigmodperc', 'fthigmotperc', 'flean', 'frear', 'fgroomfr', 'fgroomdur', 'fgroombout', 'fjump', 'fdefec'])          
females = np.array(df2.as_matrix(columns = None), dtype=bool).astype(np.int)

X = females
pca = PCA(n_components=2)
pca.fit(X)
PCA(copy=True, n_components=2, whiten=False)

#print(pca.explained_variance_ratio_) 
#print(pca.explained_variance_)

#Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

#import pylab as pl
#pl.scatter(X[:, 0], X[:, 3], c='r')
#pl.show()
