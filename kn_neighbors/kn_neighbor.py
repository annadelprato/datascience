#kn_neighbor.py
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
 
#adapted from http://blog.yhathq.com/posts/classification-using-knn-and-python.html

'''This script gets the job done but the accuracy can be improved.'''

df = pd.read_csv('iris_test.csv', names=['slength', 'swidth', 'plength', 'pwidth', 'species'], header=0)
df.dropna(inplace=True)

#print df

print df.describe()
# Scatter matrix with histograms for visualization
from pandas.tools.plotting import scatter_matrix
scatter_matrix(df, figsize=[12,8])
plt.show()

#Randomly sample 60% training data, 40% testing data
test_idx = np.random.uniform(0, 1, len(df)) <= 0.4
train = df[test_idx==True]
test = df[test_idx==False]
 
features = ['slength', 'swidth', 'plength', 'pwidth']
 
results = []
for n in range(1, 50, 2):
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(train[features], train['pwidth'])#Petal width was chosen because this trait had the greatest variation among the four species.
    preds = clf.predict(test[features])
    #If True = 1, if False = 0
    accuracy = np.where(preds==test['pwidth'], 1, 0).sum() / float(len(test))
    print "Neighbors: %d, Accuracy: %3f" % (n, accuracy)
 
    results.append([n, accuracy])

 #Results stored in a pandas dataframe for visualization and training.
results = pd.DataFrame(results, columns=["n", "accuracy"])
 
 #Plot of the accuracy shows at best 35%.
plt.plot(results.n, results.accuracy)
plt.title("Accuracy with Increasing K")
plt.show() 

results = []
for w in ['uniform', 'distance', lambda x: np.log(x)]:
    clf = KNeighborsClassifier(3, weights=w)
    w = str(w)
    clf.fit(train[features], train['pwidth'])
    preds = clf.predict(test[features])
    accuracy = np.where(preds==test['pwidth'], 1, 0).sum() / float(len(test))
    print "Weights: %s, Accuracy: %3f" % (w, accuracy)
 
    results.append([w, accuracy])

results = pd.DataFrame(results, columns=["weight_method", "accuracy"])
print results    

#Classifyiing iris flower family based on features.
new_flower_features = [[5.0, 3.0, 4.0, 1.2]]#Iris-Versicolor
#new_flower_features = [[2.0, 1.0, 1.0, 0.2]]#Iris-Setosoa
clf = KNeighborsClassifier(10, 'distance')
clf.fit(train[features], train['species'])
print clf.get_params()
print clf.predict(new_flower_features)