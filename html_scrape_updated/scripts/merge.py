import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

#Reading in the csv files and dropping the missing values.

df=pd.read_csv('gdp.csv', delimiter=',')
df.dropna(inplace=True)
print df

df1=pd.read_csv('education.csv', delimiter=',')
df1.dropna(inplace=True)
print df1

Left=df1
Right=df

#The merge operation using the 'outer' option which uses the union of keys from both data frames. The area/country columns were merged
result=pd.merge(Left, Right, on='area', how='outer', sort=True, suffixes=('_x', '_y'))
result.dropna(inplace=True)

print result

result.to_csv('merge.csv')
