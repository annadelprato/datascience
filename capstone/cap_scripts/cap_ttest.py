from scipy.stats import ttest_ind
import pandas as pd
import numpy as np
from scipy.stats import stats
#http://docs.scipy.org/doc/scipy/reference/tutorial/stats.html#t-test-and-ks-test

df1=pd.read_csv('ofmale.csv', names=['mstrain', 'mfreq', 'mlocom', 'mthigmodperc', 'mthigmotperc', 'mlean', 'mrear', 'mgroomfr', 'mgroomdur', 'mgroombout', 'mjump', 'mdefec'], header=0)          
df1.dropna(inplace=True)
#df1.drop([df1.columns[0]], axis=1, inplace=True)	
df1.drop(df1.columns[[0, 1, 9]], axis=1, inplace=True)
print df1
df1 = np.array(df1.as_matrix(columns = None), dtype=float)
print df1

df2=pd.read_csv('offemale.csv', names=['fstrain', 'ffreq', 'flocom', 'fthigmodperc', 'fthigmotperc', 'flean', 'frear', 'fgroomfr', 'fgroomdur', 'fgroombout', 'fjump', 'fdefec'], header=0)          
df2.dropna(inplace=True)
#df2.drop([df2.columns[0]], axis=1, inplace=True)
df2.drop(df2.columns[[0, 1, 9]], axis=1, inplace=True)	

print df2
df2 = np.array(df2.as_matrix(columns = None), dtype=float)
print df2

results = stats.ttest_ind(df1,df2)#4-rearing,5-grooming frequency,8-defecation frequency 
print ('t-statistic and p-value for each column:') 
print ('locomotion, thigmotaxis(distance %), thigmotaxis(time %), leaning, rearing, grooming, grooming_duration, jumping, defection')
print results

#This is a two-sided test for the null hypothesis that 2 independent samples have identical average (expected) values. This test assumes that the populations have identical variances.
#The test measures whether the average (expected) value differs significantly across samples. If we observe a large p-value, for example larger than 0.05 or 0.1, then we cannot reject the null hypothesis of identical average scores. 
#If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%, then we reject the null hypothesis of equal averages.

#genetic basis for rearing (later in session - already habituation) and leaning against wall.
#observations motor activity filmed ethovision program measures distance/center
#observer with keyboard tallying other movements