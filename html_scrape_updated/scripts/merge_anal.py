import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

df2=pd.read_csv('data_2008.csv')
df2.dropna(inplace=True)
#print df2

df2['area'].astype(str)
df2['total'].astype(int)
df2['_2008'].astype(float)

#import math
#result = math.log10('_2008')
_2008=df2['_2008']
result = np.log10(_2008)
print result

plt.scatter(df2['total'], df2['_2008'])
plt.show()
plt.savefig("scatter_2008.png")

df3=pd.read_csv('data_2009.csv')
df3.dropna(inplace=True)
#print df2

df3['area'].astype(str)
df3['total'].astype(int)
df3['_2009'].astype(float)

#import math
#result = math.log10('_2009')

_2009=df3['_2009']
result = np.log10(_2009)#log transform
print result

plt.scatter(df3['total'], df3['_2009'])
plt.show()
plt.savefig("scatter_2009.png")

#The good, the bad and the ugly

#bad & ugly
#Scraping in general went ok  I was able to isolate the necessary data but could not export it for further manipulation. Also, it was not clear what to reach for.
#I have problems getting the log transforrm via Python math. It works with Numpy but then you are working in an array and not a dataframe. I think the problem was related to floats 
#I have  problems with formatting/prepping the csv files for analysis.
#I was not able to select a subset of the data in a dataframe for plotting. I would like to learn this.

#the good:)

#To answer the question as to whether a relationship exists between GDP and number of years in school per country, I merged the two data sets (merge.py)
#and then plotted the data for the years 2008 and 2009 which represented 62% of the total data.

#The merge operation was verified.

#Based on the plotted GDP_2009/total years in school for 27 countries, it looks as though a relationship between the two variables exists.
#Based on the plotted GDP_2008/total years in shcool for 62 countries, it looks as though a relationship between the two variables exists.
