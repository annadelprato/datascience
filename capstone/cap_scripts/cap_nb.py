import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('openfield.csv', names=['locom', 'thigmodperc', 'thigmotperc', 'lean', 'rear', 'groomfr', 'groomdur', 'groombout', 'jump', 'defec', 'gender'], header=0) 

#map sex to categorical variable and checking
A = pd.Categorical(df['gender'].tolist())
print(len(A[A=='male']))#55 strain means
print(len(A[A=='female']))#54 strain means

data = df[['groomfr', 'rear', 'defec']]
target = A
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = (gnb.fit(data, target).predict(data))
print("Number of mislabeled points out of a total %d points: %d" %(data.shape[0], (target != y_pred).sum()))
'''Number of misclassified/mislabeled points are 35 out of a total of 109 points ~32 percent of the data.'''
#predict the gender given a set of criteria
#d = {'groomfr': 5, 'rear': 5, 'defec': 3 } #male
d = {'groomfr': 3, 'rear': 3, 'defec': 1 } #female
df = pd.DataFrame(data=d, index=[1])
pred = gnb.fit(data,target).predict(df)
print(pred) 

d = {'groomfr': 5, 'rear': 5, 'defec': 3 }
df = pd.DataFrame(data=d, index=[1])
pred = gnb.fit(data,target).predict(df)
print(pred)

#male 
#female

#Naive Bayes - independent measures small sample sizes.
#Naive Bayes classifier will take an observation, compute the Bayesian probability of each possible hypothesis, 
#and select the hypothesis with the highest probability (the Maximum A Posteriori (MAP) hypothesis) to classify the observation.
#A Gaussian Naive Bayes classifier assumes that the likelihood of the features is assumed to be Gaussian.
#http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

#A set of parameter/conditions could be found where the NB prediction works but it's fuzzy - can classify but perhaps, not predict well for this data.
#~1/3 of the data not used