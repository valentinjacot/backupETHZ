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
    count   = np.zeros(int(nbins))
    bins    = np.linspace(min_val, max_val, num = nbins)
    for x in xarr:
        bin_number = int((nbins-1) * ((x - min_val) / (max_val - min_val)))
        count[bin_number] += 1
   
    """
    TODO:
        Task 2.b - add your changes here (remove this comment)
        ...
        
    """
    
    return count, bins

"""
TODO:
    Task 2.a - play with param numbins and study the full data set

"""
numbins = 5
counts, bins = hist(data,numbins)



###############################################################################
### Subtask 2. Visualise Data
### Subtask 3. Visualise Data
###############################################################################

plt.bar(bins, counts, width=0.5, align='edge')
plt.show()  # Hint: you might want to uncomment this line as you advance with
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



###############################################################################
### Subtask 8. Visualisation
###############################################################################

"""
TODO:
    Subtask 8. - Plot the density function of your ditribution and the Gaussian.

    Following the examples of the Gaussian distribution this could look like:
    
    x = np.arange(0, data.max())
    gdensity = list(map(lambda d: lk_gaussian([d], muHat, varHat), x))
    plt.plot(x, gdensity)
    plt.show()

"""


