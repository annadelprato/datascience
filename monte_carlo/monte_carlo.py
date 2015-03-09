#This script demonstrates simulations of coin flipping
#creates 1000 simulated coins, flips all 100 times, prints # of heads
import random
import numpy as np
import matplotlib.pyplot as plt

# let's create a fair coin object that can be flipped:

class Coin(object):
    '''this is a simple fair coin, can be pseudorandomly flipped'''
    sides = ('heads', 'tails')
    last_result = None

    def flip(self):
        '''call coin.flip() to flip the coin and record it as the last result'''
        self.last_result = result = random.choice(self.sides)
        return result

# let's create some auxilliary functions to manipulate the coins:

def create_coins(number):
    '''create a list of a number of coin objects'''
    return [Coin() for _ in xrange(number)]

def flip_coins(coins):
    '''side effect function, modifies object in place, returns None'''
    for coin in coins:
        coin.flip()

def count_heads(flipped_coins):
    return min(coin.last_result == 'heads' for coin in flipped_coins)

def count_tails(flipped_coins):
    return max(coin.last_result == 'tails' for coin in flipped_coins)


def main():
    coins = create_coins(1000)
    for i in xrange(100):
        flip_coins(coins)
        print count_heads(coins)
                   
if __name__ == '__main__':
    main()
   
#I imported the coin toss results as a csv file for plotting. 
#The hsitogram shows that the coin toss data has a 'normal' distribution.
data=np.loadtxt(open("coins.csv","rb"),delimiter=",",skiprows=1)
print data
print data.min()
print data.max()

plt.hist(data, histtype='bar')
plt.title('coin toss')
plt.show()

#I generated a normal variable/distribution via numpy.
mu, sigma = 0, 0.1
s = np.random.normal(mu, sigma, 1000)
#print s
#print s.shape#1000L,
#print s.dtype#float64

count, bins, ignored = plt.hist(s, 30, normed=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.title('normal variable')
plt.show()

#s = np.array( [range(1000)] )
#print s.min()
#print s.max()
#print s
#s = np.arange(1000).reshape(2, 500)
#print s.shape# (1L, 1000L)
#print s.ndim# 2

#Attempts to modify the program in order to track the max and min of each trial of the normal variable /100 simulations and commment on the distribution tails.
#This code generates a lot of numbers but doesn't yet do what it needs to.
#for x1 in np.nditer(s):
   # if x1.min() < 10:
        #print x1.min()  
        #print([x1.min() for _ in xrange(100)])#or 100 etc
             
#for x2 in np.nditer(s):
    #if x2.max() > 990:
        #print x2.max()   
        #print([x2.max() for _ in xrange(100)])    

#for i in xrange(s):
    #print '%04.3f' % random.uniform(1, 10)   
    #min + (max - min) * random() 
    
#print '[1, 10]:'
#for i in xrange(100):
    #print random.randint(1, 100)

'''I am having difficulty with this exercise and I'm still working on it but I have submitted my work to this point due to the course ending date. 
I think that plotting the results from a simulation which records the min and max of a normal variable 
i.e. the tails of the distibution, would have a bimodal trace/distribution. The distribution of all the min and all the max respectively,
would eventually.... converge to normal distribution. '''
