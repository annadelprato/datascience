import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

#loansData = pd.read_csv('loansData.csv')

#print the first 5 rows of each of the column to see what needs to be cleaned
print loansData['Interest.Rate'][0:5]
print loansData['Loan.Length'][0:5]
print loansData['FICO.Range'][0:5]

#cleaning up the columns
loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: x.rstrip('%'))
loansData['Loan.Length'] = loansData['Loan.Length'].map(lambda x: x.rstrip('months'))

#printing again to see if cleaning took place or not
print loansData['Interest.Rate'][0:5]
print loansData['Loan.Length'][0:5]

'''convert the data in FICO.Range into string and split the string and take the lowest value'''
loansData['FICO.Score'] = loansData['FICO.Range'].astype(str)
print loansData['FICO.Score'][0:5]
A =loansData['FICO.Score'].tolist()
#print (A)
FICO=[] #declare an empty array
for j in range(len(A)): #for j in between 0 to len(A)
   B = A[j].split("-") #split each sub-array on - and save it to B
   #C = int(B[0], B[1]) #convert the str to int
   #C = np.mean(C) #finding the mean of B[0] and B[1]
   C = int(B[0]) #convert the string to int, using only the first value
   FICO.append(C) #append each C to the empty array, using first value
loansData['FICO.Score']=FICO

#plot histogram
plt.figure()
p=loansData['FICO.Score'].hist()
plt.show()

#create a scatterplot matrix
#a=pd.scatter_matrix(loansData, alpha=0.05, figure=(10,10))
#plt.show()
a=pd.scatter_matrix(loansData, alpha=0.05, figure=(10,10), diagonal='hist')
plt.show()