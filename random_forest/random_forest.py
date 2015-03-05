
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.grid_search import GridSearchCV 
from sklearn import preprocessing 
from sklearn import metrics as skm


samsungData = pd.read_csv('samdat.csv', low_memory=False)

samsungData['dxy'] = (samsungData['tGravityAcc-mean()-X'] - samsungData['tGravityAcc-mean()-Y'])
samsungData['dxz'] = (samsungData['tGravityAcc-mean()-X'] - samsungData['tGravityAcc-mean()-Z'])
samsungData['dyz'] = (samsungData['tGravityAcc-mean()-Y'] - samsungData['tGravityAcc-mean()-Z'])


#label column name
labels = 'activity'
#subject column name
subject = 'subject'
#features column name (all columns except subject and labels
features = filter(lambda x: x not in list([labels, subject]), samsungData.columns)

#labels to integer
le = preprocessing.LabelEncoder()
le.fit(samsungData[labels])

#Split the data into training and test dataset
test_subjects = [27, 28, 29, 30] # per instructions
train_subjects = filter(lambda x: x not in test_subjects, np.unique(samsungData['subject']))

# take training and test set from requested subjects
train = samsungData[samsungData['subject'].isin(train_subjects)]
test = samsungData[samsungData['subject'].isin(test_subjects)]

# training data
X = train[features]
y = le.transform(train[labels])

# test data
Xt = test[features]
yt = le.transform(test[labels])

#training with 3-fold cv (Uncomment at your own risk. It runs sloooowly.)
#train random forest classifier
#set the parameters by cross-validation
#tuned_parameters = [{'max_features': ['sqrt', 'log2'], 'n_estimators': [100, 200, 500]}]
#rf = GridSearchCV( RandomForestClassifier(min_samples_split=1, compute_importances=False, n_jobs=-1), tuned_parameters, cv=3, verbose=2 ).fit(X, y)
#print 'Best parameters set found on development set:'
#print rf.best_estimator_
#pred = rf.predict(Xt)
#print 'prediction accuracy: %.4f' % (1 - (1. / len(yt) * sum( pred != yt )))

import matplotlib.pyplot as plt
#testing
#http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

rfc = RandomForestClassifier(max_features='log2', n_estimators=500, min_samples_split=1, compute_importances=True, n_jobs=-1).fit(X, y)
rfc.feature_importances_.shape
number_of_features = 10
indices = np.argsort(rfc.feature_importances_)[::-1][:number_of_features]
#std = np.std([rfc.feature_importances_ ], axis=0) for error bars - not working yet

#Plot the feature importances 
#plt.figure()
plt.title("Feature importances")
plt.bar(xrange(number_of_features), rfc.feature_importances_[indices], color="r")#, yerr=std[indices], align="center") for error bars - not working yet 
plt.xticks(xrange(number_of_features), X.columns[indices], rotation='vertical', fontsize=12)
plt.xlim([-1, number_of_features])
plt.show()

clf = RandomForestClassifier(n_estimators=500, oob_score=True).fit(X, y)
# predict on training data just for testing
y_predict = clf.predict(X)

# OOB score 
print 'OOB score: %.2f\n' % clf.oob_score_ #0.99

# confusion matrix on training data
from sklearn.metrics import confusion_matrix
print 'Confusion matrix:'
print skm.confusion_matrix(y, y)
cm = confusion_matrix(y, y)
#print(cm)

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Accuracy
print("Accuracy = %f" %(skm.accuracy_score(yt, yt)))#1.0
print("Accuracy = %f" %(skm.accuracy_score(y, y)))#1.0

# Precision
print("Precision = %f" %(skm.precision_score(yt, yt)))#1.0
print("Precision = %f" %(skm.precision_score(y, y)))#1.0

# Recall
print("Recall = %f" %(skm.recall_score(yt, yt)))#1.0
print("Recall = %f" %(skm.recall_score(y, y)))#1.0

# F1 Score
print("F1 score = %f" %(skm.f1_score(yt, yt)))#1.0
print("F1 score = %f" %(skm.f1_score(y, y)))#1.0

print("mean accuracy score for testing set = %f" %(rfc.score(Xt, yt)))#0.960943
print("mean accuracy score for training set = %f" %(rfc.score(X, y)))#1.0
