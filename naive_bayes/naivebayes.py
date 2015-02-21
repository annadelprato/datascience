
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ideal_weight.csv', names=['id', 'sex', 'actual', 'ideal', 'diff'], header=0)
print df

#file cleanup
df['sex']=df['sex'].map(lambda x: x.replace("'",""))
#df['sex']=df['sex'].map(lambda x: x.rstrip('').lstrip('')
A=df['sex'].tolist()
B=[]
for i in range(len(A)):
    test=A[i].strip(' ')
    B.append(test)
df['sex']=B

print df

#Plots of actual and ideal weight.
plt.hist([df['actual'], df['ideal']], histtype='bar', stacked=True)
plt.title('Actual and Ideal Weight')
plt.show()
plt.savefig("weights.png")

#Plots the distribution of weight difference.
plt.hist(df['diff'], histtype='bar')
plt.title('Weight Difference')
plt.show()
plt.savefig("wt_diff.png")

#Mapping sex to categorical variable indicates that there are 63 males and 119 females in the dataset.
A = pd.Categorical(df['sex'].tolist())
print(len(A[A=='Male']))
print(len(A[A=='Female']))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

#Fitting a Naive Bayes classifier of sex to actual weight, ideal weight, and diff.
data = df[['actual', 'ideal', 'diff']]
target = A

y_pred = gnb.fit(data, target).predict(data)
print("Number of mislabeled points out of a total %d points: %d" %(data.shape[0], (target != y_pred).sum()))
#Number of mislabeled points is 14 from 182 total points. 
#Predicts the gender for a weight of 145, ideal weight of 160, and a difference of -15.
d = {'actual': 145, 'ideal': 160, 'diff': -15}
df = pd.DataFrame(data=d, index=[1])
pred = gnb.fit(data,target).predict(df)
print ("The model describes a")
print(pred)
#The model predicts that the data describes a male.