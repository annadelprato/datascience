from sklearn import datasets
iris = datasets.load_iris()
import numpy as np

import matplotlib.pyplot as plt
plt.scatter(iris.data[:, 1], iris.data[:, 2], c=iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.title('Iris dataset')
plt.show()

#The first 100 observations correspond to setosa and versicolor.
plt.scatter(iris.data[0:100, 1], iris.data[0:100, 2], c=iris.target[0:100])
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.show()


from sklearn import svm
svc = svm.SVC(kernel='linear')
from sklearn import datasets

#SVC model for two species of iris flower.
X = iris.data[0:100, 1:3]
y = iris.target[0:100]
svc.fit(X, y)

#SVC model for three species of iris flower.
X = iris.data[0:, 1:3]
y = iris.target[0:]
svc.fit(X,y)

#Adapted from https://github.com/jakevdp/sklearn_scipy2013
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

#def plot_estimator(estimator, X, y):
    #estimator.fit(X, y)
    #x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    #y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    #xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
    #Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])   

    # Output the above result to a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()

plot_estimator(svc, X, y)
plt.show()

'''Representative plots are shown.'''
'''Changing the meshgrid values effect the quality of the trace as well as the boundaries.'''
'''Setosa and Versicolor for the petal length and sepal width traits partition cleanly.'''
'''If correct, there is not much difference in the code to select 3 versus 2 iris species for the svm classification.''' 