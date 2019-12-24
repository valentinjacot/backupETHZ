#!/usr/bin/env python
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
import matplotlib.pyplot as plt

# class defining the posterior distribution for the coin flip example
class target_coin_flip():
    def __init__(self, NN, NH):
        self.NN = NN # number of tosses
        self.NH = NH # number of heads
    def evaluate(self, H):
        # samples from the posterior
        pH=1 if (H>=0 and H<=1) else 0.0
        return np.power(H,self.NH)*np.power(1-H,self.NN-self.NH)*pH

#class defining the logarithmic representation of the posterior
class target_log_coin_flip():
    # TODO : use the "target_coin_flip" class as a starting point
    #        and edit appropriately to convert to logarithmic scale


def MCMC(target, starting_sample, num_iters=1e6, burnin=1e4):
    print("Running MCMC")
    # Markov Chain Monte Carlo (MCMC) - Metropolis
    current_sample = starting_sample
    # proposal distribution is a Gaussian with mean 0.0 and std 0.1
    proposal_mean = 0.0
    proposal_sigma = 0.1
    samples = []
    samples.append(starting_sample)
    for iter_ in range(int(num_iters)):
        # GENERATION BASED ON PROPOSAL
        next_sample_candidate = current_sample + proposal_mean + np.random.randn()*proposal_sigma
        # ACCEPTANCE PROBABILITY - Metropolis
        acceptance_prob = np.min([1.0, target.evaluate(next_sample_candidate)/(target.evaluate(current_sample))])
        # ACCEPT OR REJECT with uniform probability
        temp = np.random.rand()
        if temp <= acceptance_prob:
            # ACCEPTING THE CANDIDATE SAMPLE
            samples.append(next_sample_candidate)
            current_sample = next_sample_candidate
        else:
            samples.append(current_sample)
    if len(samples)>burnin:
        # keep only samples after burn-in iterations
        samples=samples[int(burnin):]
    else:
        raise ValueError("Number of samples {:} smaller than burnin period {:}.".format(len(samples), burnin))
    return samples


def MCMCLOG(logtarget, starting_sample, num_iters=1e6, burnin=1e4):
    print("Running MCMCMHLOG")
    # Markov Chain Monte Carlo (MCMC) - Metropolis Hastings 
    # TODO: start from the "MCMC" function above and edit whenever
    #       necessary to convert to logarithmic scale


if __name__ == "__main__":

    #IN NN TOSSES, NH TIMES HEAD (NN>=NH)
    NN = 300 # NN = 300 / 3000 tosses
    NH = 150 # NH = 150 / 1500 heads
    burnin = 1e2 # number of burn-in iterations
    num_iters = 1e5 # number of total MCMC iterations
    starting_sample = 0.5 # starting point for MCMC algorithm
    
    target = target_coin_flip(NN,NH)
    samples_mcmc = MCMC(target, starting_sample, num_iters, burnin)
    # plot histogram with samples drawn from posterior distribution with the MCMC algorithm
    plt.hist(samples_mcmc, normed=True, facecolor='g', alpha=0.6, label="MCMC") # if you're using python>3, use density=True instead of normed=True
    plt.xlim([0,1])
    plt.legend()
     # plt.show()

    target = target_log_coin_flip(NN,NH)
    samples_mcmclog = MCMCLOG(target, starting_sample, num_iters, burnin)
    plt.hist(samples_mcmclog, normed=True, facecolor='b', alpha=0.6, label="MCMC-LOG")
    plt.xlim([0,1])
    plt.legend()
    plt.show()








