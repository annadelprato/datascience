import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

#df=pd.read_csv('merge.csv')
df=pd.read_csv('merge2.csv')
df.dropna(inplace=True)
country = df['Area']
print df[country]#still gives key errors self get item etc
#a = pd.scatter(df, alpha=0.05, figsize=(10,10))
#plt.show()

#linear model example
#country = df['Area']
#total = df['Total']
#year = df['1999']

# The dependent variable
#y = np.matrix(total).transpose()
# The independent variables shaped as columns
#x1 = np.matrix(country).transpose()
#x2 = np.matrix(year).transpose()

#x = np.column_stack([x1,x2])

#X = sm.add_constant(x)
#model = sm.OLS(y,X)
#f = model.fit()

#print 'Coefficients: ', f.params[0:2]
#print 'Intercept: ', f.params[2]
#print 'P-Values: ', f.pvalues
#print 'R-Squared: ', f.rsquared