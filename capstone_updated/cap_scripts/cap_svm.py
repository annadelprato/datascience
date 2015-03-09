import pandas as pd
import numpy as np
from sklearn import svm
svc = svm.SVC(kernel='linear')

df1=pd.read_csv('openfield2.csv')
A = pd.Categorical(df1['labels'].tolist())
df1.loc[df1['labels'] == 'male', 'labels'] = 0
df1.loc[df1['labels'] == 'female', 'labels'] = 1
#print df1
df1 = np.array(df1.as_matrix(columns = None), dtype=float)
#print df1

#SVC model for two factors.

X = df1[0:100, 4:6]#choosing columns 4 rearing & 5 grooming frequency, leaving out the last 10 data points
y = df1[0:100, 10]
svc.fit(X, y)

import matplotlib.pyplot as plt


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

