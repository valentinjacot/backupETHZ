### HPCSE II Spring 2019
### HW 2 - Task 3: Data Analysis

###############################################################################
### Import Modules
###############################################################################
import numpy as np
import matplotlib.pyplot as plt



###############################################################################
### Generate Data
###############################################################################

numSamples = 20193
mean       = 3.14

np.random.seed(1337)

data0 = np.random.poisson(mean, numSamples)
np.save('task4.npy', data0)



###############################################################################
### Subtask 1. Read Data
###############################################################################

data = np.load('task4.npy') # full data set

#data = data[1:100] # Hint: use this line for testing purpose 



###############################################################################
### Subtask 2. Histogram
###############################################################################

def hist(xarr, nbins):
    min_val = xarr.min()
    max_val = xarr.max()
    count   = np.zeros(int(nbins))
    bins    = np.linspace(min_val, max_val, num = nbins)
    for x in xarr:
        bin_number = int((nbins-1) * ((x - min_val) / (max_val - min_val)))
        count[bin_number] += 1
   
    count/=sum(count)
    
    return count, bins


numbins = max(data)+1
counts, bins = hist(data,numbins)



###############################################################################
### Subtask 2. Visualise Data
### Subtask 3. Visualise Data
###############################################################################

plt.bar(bins, counts, width=0.5, color='grey')
#plt.show()  # Hint: you might want to uncomment this line as you advance with
             # the exercise in order to avoid interuptions



###############################################################################
### Subtask 4. (nothing to do here - only on paper)
###############################################################################



###############################################################################
### Subtask 5. Likelihood and Log-likelihood
###############################################################################

## more imports
from operator import mul
from functools import reduce


## Lambda functions for Poisson likelihood and log-likelihood
lk_poisson  = lambda x, mu: 1/reduce(mul , map(np.math.factorial,x))*mu**np.sum(x)*np.exp(-len(x)*mu)
llk_poisson = lambda x, mu: -len(x)*mu + np.sum(x)*np.log(mu)-sum(np.log( list(map(np.math.factorial,x)) ))



###############################################################################
### Subtask 6. Distribution function
###############################################################################

muHat = np.mean(data) # muHat counts for both, Poisson and Gaussian
print(muHat)



###############################################################################
### Subtask 7. Comparison with Gaussian Distribution
###############################################################################

## calculate..
lkp  = lk_poisson(data, muHat)
llkp = llk_poisson(data, muHat)


## generate output ..
print("Poisson likelihood: " + str(lkp))
print("Poisson loglikelihood: " + str(llkp))
print("exp(loglikelihood): " + str(np.exp(llkp)))


# Lambda functions for Gaussian likelihood and loglikelihood
lk_gaussian  = lambda x, mu, var: (2*np.math.pi*var)**(-0.5*len(x)) * np.exp(-0.5 * sum((x-mu)**2) /var)
llk_gaussian = lambda x, mu, var: -0.5*len(x)*np.log(2*np.math.pi)-0.5*len(x)*np.log(var)-0.5*sum((x-mu)**2)/var


# MLE estimator for Gaussian variance 
varHat = np.var(data)


# calculate..
lkg = lk_gaussian(data, muHat, varHat)
llkg = llk_gaussian(data, muHat, varHat)


# generate output ..
print("Gaussian likelihood: " + str(lkg))
print("Gaussian loglikelihood: " + str(llkg))
print("exp(loglikelihood): " + str(np.exp(llkg)))



###############################################################################
### Subtask 8. Visualisation
###############################################################################

x = np.arange(0, max(data))

py = list(map(lambda d: lk_poisson([d], muHat), x))
gy = list(map(lambda d: lk_gaussian([d], muHat, varHat), x))

plt.plot(x, gy, label='Gaussian distribution', color = 'orange')
plt.stem(x, py, label='Poisson distribution')
plt.xlabel("samples")
plt.xticks(np.arange(0, 12, step=1))
plt.ylabel("$pdf(x=x_i)$")
plt.legend(loc='upper right')
plt.show()



###############################################################################
### END
###############################################################################
