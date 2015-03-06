import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df1=pd.read_csv('ofmale.csv', names=['mstrain', 'mfreq', 'mlocom', 'mthigmodperc', 'mthigmotperc', 'mlean', 'mrear', 'mgroomfr', 'mgroomdur', 'mgroombout', 'mjump', 'mdefec'], header=0)          
df1.dropna(inplace=True)


df2=pd.read_csv('offemale.csv', names=['fstrain', 'ffreq', 'flocom', 'fthigmodperc', 'fthigmotperc', 'flean', 'frear', 'fgroomfr', 'fgroomdur', 'fgroombout', 'fjump', 'fdefec'], header=0)
df2.dropna(inplace=True)
#print df2	                                    
#print df2.shape


males = df1.as_matrix(columns=['mstrain', 'mfreq', 'mlocom', 'mthigmodperc', 'mthigmotperc', 'mlean', 'mrear', 'mgroomfr', 'mgroomdur', 'mgroombout', 'mjump', 'mdefec']);
females = df2.as_matrix(columns=['fstrain', 'ffreq', 'flocom', 'fthigmodperc', 'fthigmotperc', 'flean', 'frear', 'fgroomfr', 'fgroomdur', 'fgroombout', 'fjump','fdefec']);
	                                     

# scatter plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(males[:,2],females[:,2], s=100, c=['b', 'y'])
plt.xlabel('males')
plt.ylabel('females')
plt.title('Total Distance Traveled')
plt.show()
#plt.savefig("locum_scat.png")

# scatter plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(males[:,3],females[:,3], s=100, c=['b', 'y'])
plt.xlabel('males')
plt.ylabel('females')
plt.title('Distance from Periphery (arbitrary units)')
plt.show()
#plt.savefig("thigmod_scat.png")

# scatter plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(males[:,4],females[:,4], s=100, c=['b', 'y'])
plt.xlabel('males')
plt.ylabel('females')
plt.title('Time at Periphery (sec)')
plt.show()
#plt.savefig("thigmot_scat.png")

# scatter plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(males[:,5],females[:,5], s=100, c=['b', 'y'])
plt.xlabel('males')
plt.ylabel('females')
plt.title('Leaning Frequency')
plt.show()
#plt.savefig("lean_scat.png")

# scatter plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(males[:,6],females[:,6], s=100, c=['b', 'y'])
plt.xlabel('males')
plt.ylabel('females')
plt.title('Rearing Frequency')
plt.show()
#plt.savefig("rear_scat.png")

# scatter plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(males[:,7],females[:,7], s=100, c=['b', 'y'])
plt.xlabel('males')
plt.ylabel('females')
plt.title('Grooming Frequency')
plt.show()
#plt.savefig("groomfr_scat.png")

# scatter plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(males[:,8],females[:,8], s=100, c=['b', 'y'])
plt.xlabel('males')
plt.ylabel('females')
plt.title('Grooming Duration')
plt.show()
#plt.savefig("groomdur_scat.png")

# scatter plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(males[:,9],females[:,9], s=100, c=['b', 'y'])
plt.xlabel('males')
plt.ylabel('females')
plt.title('Grooming Bout')
plt.show()
#plt.savefig("groombout_scat.png")

# scatter plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(males[:,10],females[:,10], s=100, c=['b', 'y'])
plt.xlabel('males')
plt.ylabel('females')
plt.title('Jumping Frequency')
plt.show()
#plt.savefig("jump_scat.png")

# scatter plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(males[:,11],females[:,11], s=100, c=['b', 'y'])
plt.xlabel('males')
plt.ylabel('females')
plt.title('Defecation Frequency')
plt.show()
#plt.savefig("defec_scat.png")