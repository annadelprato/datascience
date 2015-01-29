import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df=pd.read_csv('htmltabletocsv.csv')
df.dropna(inplace=True)
print df

print df['Country or area']
print df['Total']
print df['Men']
print df['Women']


df.hist(column='Women')
plt.show()
#plt.savefig("hist_Boston.png")

df.hist(column='Men')
plt.show()
#plt.savefig("hist_Boston.png")


print "Women", df['Women'].describe()
print "Men", df['Men'].describe()
#print "Total", df['Total'].describe()

print df['Women'].var() 
print df['Women'].median() 
print df['Men'].var()
print df['Men'].median()