import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df1=pd.read_csv('ofmale.csv', names=['mstrain', 'mfreq', 'mlocom', 'mthigmodperc', 'mthigmotperc', 'mlean', 'mrear', 'mgroomfr', 'mgroomdur', 'mgroombout', 'mjump', 'mdefec'], header=0)
	          
df1.dropna(inplace=True)
print df1
#print df1.describe()
#print df1.shape

df2=pd.read_csv('offemale.csv', names=['fstrain', 'ffreq', 'flocom', 'fthigmodperc', 'fthigmotperc', 'flean', 'frear', 'fgroomfr', 'fgroomdur', 'fgroombout', 'fjump', 'fdefec'], header=0)
df2.dropna(inplace=True)
print df2	                                    
#print df2.shape

#stacked bar, bar also tried but less clear than distinct separate hist plots
#could this have benn streamlined? i.e., histograms generated for each column.

plt.hist(df1['mlocom'])
plt.title('Male Locomotion')
plt.show()
plt.savefig("mlocom.png")

plt.hist(df2['flocom'])
plt.title('Female Locomotion')
plt.show()
plt.savefig("flocom.png")

plt.hist(df1['mlean'])
plt.title('Male Leaning')
plt.show()
plt.savefig("mlean.png")

plt.hist(df2['flean'])
plt.title('Female Leaning')
plt.show()
plt.savefig("flean.png")

plt.hist(df1['mrear'])
plt.title('Male Rearing')
plt.show()
plt.savefig("rearing.png")

plt.hist(df2['frear'])
plt.title('Female Rearing')
plt.show()
plt.savefig("frearing.png")

plt.hist(df1['mthigmodperc'])
plt.title('Male Periphery (distance)')
plt.show()
plt.savefig("mperiphery.png")

plt.hist(df2['fthigmodperc'])
plt.title('Female Periphery (distance)')
plt.show()
plt.savefig("fperiphery.png")

plt.hist(df1['mthigmotperc'])
plt.title('Male Periphery (time)')
plt.show()
plt.savefig("mperiphery.png")

plt.hist(df2['fthigmotperc'])
plt.title('Female Periphery (time)')
plt.show()
plt.savefig("fperiphery.png")

plt.hist(df1['mgroomfr'])
plt.title('Male Grooming (frequency)')
plt.show()
plt.savefig("mgroomfr.png")

plt.hist(df2['fgroomfr'])
plt.title('Female Grooming (frequency)')
plt.show()
plt.savefig("fgroomfr.png")

plt.hist(df1['mgroombout'])
plt.title('Male Grooming Bout')
plt.show()
plt.savefig("mgroombout.png")

plt.hist(df2['fgroombout'])
plt.title('Female Grooming Bout')
plt.show()
plt.savefig("fgroombout.png")

plt.hist(df1['mjump'])
plt.title('Male Jumping')
plt.show()
plt.savefig("mjumping.png")

plt.hist(df2['fjump'])
plt.title('Female Jumping')
plt.show()
plt.savefig("fjumping.png")

plt.hist(df1['mdefec'])
plt.title('Male Defecation')
plt.show()
plt.savefig("mpoop.png")

plt.hist(df2['fdefec'])
plt.title('Female Defecation')
plt.show()
plt.savefig("fpoop.png")
