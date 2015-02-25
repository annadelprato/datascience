#load the data
from sklearn import datasets, svm
iris = datasets.load_iris()

X = iris.data[0:,1:3]#[:,0:2]
y = iris.target[0:]

import numpy as np

print iris.data

svc = svm.SVC(C=1, kernel='linear')

from sklearn import cross_validation
kfold = cross_validation.KFold(len(X), n_folds=5)
result=[svc.fit(X[train], y[train]).score(X[test], y[test]) for train, test in kfold]
print result #'''1.0, 0.97, 0.80, 0.77, 0.80'''
print(np.mean(result)) #'''0.906...'''
print(np.std(result)) #'''0.088...'''

#same numerical output as above.
#cv_score = cross_validation.cross_val_score(svc, X, y, cv=kfold, n_jobs=-1)
#print cv_score
#print(np.mean(cv_score))
#print(np.std(cv_score))
#print(np.mean(cv_score))
#print(np.std(cv_score))

#'''Alternate score estimator paramters
f1 = cross_validation.cross_val_score(svc, X, y, cv=kfold, scoring='f1', n_jobs=-1)
print f1
#Trying to get the f1 score gives a warning, 'UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no true samples.
#Numerical ouput: 0. 0.982... 0.889...  0.952  0.'''