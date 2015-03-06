import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df1=pd.read_csv('ofmale.csv', names=['mstrain', 'mfreq', 'mlocom', 'mthigmodperc', 'mthigmotperc', 'mlean', 'mrear', 'mgroomfr', 'mgroomdur', 'mgroombout', 'mjump', 'mdefec'], header=0)          
df1.dropna(inplace=True)


df2=pd.read_csv('offemale.csv', names=['fstrain', 'ffreq', 'flocom', 'fthigmodperc', 'fthigmotperc', 'flean', 'frear', 'fgroomfr', 'fgroomdur', 'fgroombout', 'fjump', 'fdefec'], header=0)
df2.dropna(inplace=True)
#print df2	                                    
#print df2.shape

stats1 = df1.describe()
print stats1

stats2 = df2.describe()
print stats2

var1= df1.var()
print ('variance males')
print var1

var2= df2.var()
print ('variance females')
print var2



males = df1.as_matrix(columns=['mstrain', 'mfreq', 'mlocom', 'mthigmodperc', 'mthigmotperc', 'mlean', 'mrear', 'mgroomfr', 'mgroomdur', 'mgroombout', 'mjump', 'mdefec']);
females = df2.as_matrix(columns=['fstrain', 'ffreq', 'flocom', 'fthigmodperc', 'fthigmotperc', 'flean', 'frear', 'fgroomfr', 'fgroomdur', 'fgroombout', 'fjump','fdefec']);

#Spearman correlation matrix within dataframe '''grooming frequency & defecation'''somewhat high correlation - moreso for females - indicator of stress
result2=df1.corr()
print result2

#Spearman correlation matrix within dataframe
result3=df2.corr()
print result3

#Spearman correlation matrix between dataframes column by column /Spearman not effected by outliers.
# Again, couldn't find a way to streamline this...

print 'Spearman Rank correlations'

result1 = df1['mlocom'].corr(df2['flocom'], method='spearman')
print result1 #0.79...

result2 = df1['mthigmotperc'].corr(df2['fthigmotperc'], method='spearman')
print result2 #0.77...

result3 = df1['mthigmodperc'].corr(df2['fthigmodperc'], method='spearman')
print result3 #0.80...

result4 = df1['mlean'].corr(df2['flean'], method='spearman')
print result4 #0.84...

result5 = df1['mrear'].corr(df2['frear'], method='spearman')
print result5 #0.61...

result6 = df1['mgroomfr'].corr(df2['fgroomfr'], method='spearman')
print result6 #0.61...

result7 = df1['mgroomdur'].corr(df2['fgroomdur'], method='spearman')
print result7 #0.70...

result8 = df1['mgroombout'].corr(df2['fgroombout'], method='spearman')
print result8 #0.65

result9 = df1['mjump'].corr(df2['fjump'], method='spearman')
print result9 #0.71

result10 = df1['mdefec'].corr(df2['fdefec'], method='spearman')
print result10 #0.60