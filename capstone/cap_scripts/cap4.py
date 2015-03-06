import numpy as np
import pandas as pd
import statsmodels.api as sm

#regression analysis males
data=pd.read_csv('ofmale.csv')

mthigmotperc = data['mthigmotperc']
mdefec = data['mdefec']
mgroomfr = data['mgroomfr']

# linear regression
y = np.matrix(mthigmotperc).transpose()
x1 = np.matrix(mgroomfr).transpose()
X = sm.add_constant(x1)
model = sm.OLS(y,X)
f = model.fit()
print 'males thigmotperc & grooming frequency'
print(f.params)


y = np.matrix(mthigmotperc).transpose()
x1 = np.matrix(mdefec).transpose()
X = sm.add_constant(x1)
model = sm.OLS(y,X)
f = model.fit()
print 'males thigmotperc & defecation'
print(f.params)


y = np.matrix(mgroomfr).transpose()
x1 = np.matrix(mdefec).transpose()
X = sm.add_constant(x1)
model = sm.OLS(y,X)
f = model.fit()
print 'males grooming frequency & defection'
print(f.params)

# multivariate regression
y = np.matrix(mthigmotperc).transpose()
# The independent variables shaped as columns
x1 = np.matrix(mdefec).transpose()
x2 = np.matrix(mgroomfr).transpose()

x = np.column_stack([x1,x2])

X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

print 'males'
print 'Coefficients: ', f.params
print 'Intercept: ', f.params
print 'P-Values: ', f.pvalues
print 'R-Squared: ', f.rsquared

#regression analysis females
data=pd.read_csv('offemale.csv')

fthigmotperc = data['fthigmotperc']
fdefec = data['fdefec']
fgroomfr = data['fgroomfr']

# linear regression
y = np.matrix(fthigmotperc).transpose()
x1 = np.matrix(fgroomfr).transpose()
X = sm.add_constant(x1)
model = sm.OLS(y,X)
f = model.fit()
print 'females thigmotperc & grooming frequency'
print(f.params)


y = np.matrix(fthigmotperc).transpose()
x1 = np.matrix(fdefec).transpose()
X = sm.add_constant(x1)
model = sm.OLS(y,X)
f = model.fit()
print 'females thigmotperc & defecation'
print(f.params)

y = np.matrix(fgroomfr).transpose()
x1 = np.matrix(fdefec).transpose()
X = sm.add_constant(x1)
model = sm.OLS(y,X)
f = model.fit()
print 'females grooming frequency & defection'
print(f.params)

# multivariate regression
y = np.matrix(fthigmotperc).transpose()
# The independent variables shaped as columns
x1 = np.matrix(fdefec).transpose()
x2 = np.matrix(fgroomfr).transpose()

x = np.column_stack([x1,x2])

X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

print 'females'
print 'Coefficients: ', f.params
print 'Intercept: ', f.params
print 'P-Values: ', f.pvalues
print 'R-Squared: ', f.rsquared