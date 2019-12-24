### HPCSE II Spring 2019
### HW 2 - Task 4: Data Analysis

###############################################################################
### Import Modules
###############################################################################
import numpy as np
import matplotlib.pyplot as plt



###############################################################################
### Subtask 1. Read Data
###############################################################################

data = np.load('task4.npy') # full data set

#data = data[1:100] # Hint: use this line for testing purpose 



###############################################################################
### Subtask 2. Histogram
###############################################################################

def hist(xarr, nbins, continuous = True):
    min_val = xarr.min()
    max_val = xarr.max()
    size = 0
    count   = np.zeros(int(nbins))
    bins    = np.linspace(min_val, max_val, num = nbins)
    for x in xarr:
        bin_number = int((nbins-1) * ((x - min_val) / (max_val - min_val)))
        count[bin_number] += 1
        size +=1
        
    for i in range(0,nbins):
        count[i] = count[i] / size
    """
    TODO:
        Task 2.b - add your changes here (remove this comment)        
    """
		
    
    return count, bins
def hist_cumulative(xarr, nbins, continuous = True):
    min_val = xarr.min()
    max_val = xarr.max()
    size = 0
    count   = np.zeros(int(nbins))
    bins    = np.linspace(min_val, max_val, num = nbins)
    for x in xarr:
        bin_number = int((nbins-1) * ((x - min_val) / (max_val - min_val)))
        count[bin_number] += 1
        size +=1
        
    count[0] = count[0] / size    
    for i in range(1,nbins):
        count[i] = count[i] / size + count [i-1]
    """
    TODO:
        Task 2.b - add your changes here (remove this comment)        
    """
		
    
    return count, bins
"""
TODO:
    Task 2.a - play with param numbins and study the full data set

"""
numbins = 500 #5
counts, bins = hist(data,numbins)



###############################################################################
### Subtask 2. Visualise Data
### Subtask 3. Visualise Data
###############################################################################

plt.bar(bins, counts, width=0.5, align='edge')
#plt.show()  # Hint: you might want to uncomment this line as you advance with
            # the exercise in order to avoid interuptions



###############################################################################
### Subtask 4. (nothing to do here - only on paper)
###############################################################################



###############################################################################
### Subtask 5. Likelihood and Log-likelihood
###############################################################################

""" 
TODO:
    Subtask 5. - Implement two functions that calculate 
        a) the likelihood
        b) the loglikelihood

    of your distribution function. 

    For a Gaussian distribution this could look like:

    lk_gaussian  = lambda dat, mu, var: (2*np.math.pi*var)**(-0.5*len(dat)) * ...
    llk_gaussian = lambda dat, mu, var: -0.5*len(dat)*np.log(2*np.math.pi) - ...

"""
#lk_poisson = lambda dat, lam: np.power(lam, dat)*np.exp(-lam)/np.math.factorial(dat)
def lk_poisson(dat, lam):
	res = 1
	for d in dat:
		res *= lam**d * np.exp(-lam) / np.math.factorial(d)
	return res

def llk_poisson(dat, lam):
	res1 = 0
	res2 = 0
	n = dat.size
	for d in dat:
		res1 += d
		res2 += np.log(np.math.factorial(d))
	return -n*lam +  res1 * np.log(lam) - res2

###############################################################################
### Subtask 6. Distribution function
###############################################################################

"""
TODO:
    Subtask 6. - Calculate MLE(s) of the params from the data set
    a_hat = ...
    b_hat = ...
    ...

"""
def lambda_hat_poisson (dat):
	res=0
	n = len(dat)
	for d in dat:
		res += d
	return res/n

###############################################################################
### Subtask 7. Comparison with Gaussian Distribution
###############################################################################

"""
TODO:
    Subtask 7. - Calculate the likelihood and log-likelihood given the data and
    your MLE(s). Reuse your functions implemented in Subtask 5.

    For the Gaussian distribution this could look like:
    lik    = lk_gaussian(data, mu_hat, var_hat)
    loglik = llk_gaussian(data, mu_hat, var_hat)

"""
lambda_hat = lambda_hat_poisson(data)
lk = lk_poisson(data, lambda_hat)
llk = llk_poisson(data,lambda_hat) 
print("Poisson: ", lambda_hat, lk, llk, np.exp(llk))


def lk_gaussian  (dat, mu, var):
	res = 0
	for d in dat:
		res +=(d-mu)**2
	res = np.exp(-res/(2 * var))
	return res *(2*np.math.pi*var)**(-0.5*len(dat))
	 
def llk_gaussian (dat, mu, var):
	res = 0
	for d in dat:
		res +=(d-mu)**2
	return -0.5*len(dat)*np.log(2*np.math.pi*var) - 0.5 / var * res

def mu_hat_gaussian (dat):
	res=0
	for d in dat:
		res+=d
	return res /len(dat)

var_hat = lambda_hat # set var = poisson's result
mu_hat = mu_hat_gaussian(data)
lik= lk_gaussian(data, mu_hat, var_hat)
loglik = llk_gaussian(data, mu_hat, var_hat)

print("Gaussian: ",mu_hat, lik, loglik, np.exp(loglik))


###############################################################################
### Subtask 8. Visualisation
###############################################################################

"""
TODO:
    Subtask 8. - Plot the density function of your ditribution and the Gaussian.

    Following the examples of the Gaussian distribution this could look like:
    
    gdensity = list(map(lambda d: lk_gaussian([d], muHat, varHat), x))
    plt.plot(x, gdensity)
    plt.show()

"""
x = np.arange(0, data.max())
gdensity = list(map(lambda d: lk_gaussian([d], mu_hat, var_hat), x))
pdensity = list(map(lambda d: lk_poisson([d], lambda_hat), x))
plt.plot(x, gdensity, 'r+-', label='Gaussian distr.')
plt.plot(x, pdensity, 'g^-',label='Poisson distr.')
plt.title('Histogram, Poisson and Gaussian')
plt.legend()
plt.show()
