
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

#print the first 5 rows of each of the column to see what needs to be cleaned
#print df['Interest.Rate'][0:5]
#print df['Loan.Length'][0:5]
#print df['FICO.Range'][0:5]

#clean up
df['Interest.Rate'] = df['Interest.Rate'].map(lambda x: x.rstrip('%'))
df['Interest.Rate'] = df['Interest.Rate'].astype(float)
df['Loan.Length'] = df['Loan.Length'].map(lambda x: x.rstrip('months'))

#'''convert the data in FICO.Range to string. Split the string and take the lowest value'''
df['FICO.Score'] = df['FICO.Range']
print df['FICO.Score'][0:5]
A =df['FICO.Score'].tolist()
FICO=[]#empty array
for j in range(len(A)): 
  B = A[j].split("-") 
  C = float(B[0]) 
  FICO.append(C) 
df['FICO.Score']=FICO

#logistic modeling
intercept = [1] * len(df)
df['Intercept'] = intercept
# independant variables
ind_vars = ['Intercept', 'Amount.Requested', 'FICO.Score']
ir = df['Interest.Rate']
ir = [1 if x < 12 else 0 for x in ir]
df['IR_TF'] = ir
X = df[ind_vars]
y = df['IR_TF']
logit = sm.Logit(y, X)
result = logit.fit()
coeff = result.params
print coeff
'''Intercept = -60.125045
Amount Reqested = -0.000174
FICO Score = 0.087423'''