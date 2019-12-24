#!/usr/bin/env python
# # -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm, truncexpon
import matplotlib.pyplot as pyplot

Ns = 10000;

f = norm()
r = f.rvs(size=Ns)

#simple estimator
p1 = sum(r>4.5)/Ns


#importance sampling
lower, upper, scale = 4.5, np.inf, 1.0
g = truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)
r2 = g.rvs(size=Ns)

p2 = sum( (f.pdf(r2)/g.pdf(r2))*(r2>4.5) ) / Ns

exact = 1.-f.cdf(4.5)

print('\n')
print('p1 = %.10f \n' % p1)
print('p2 = %.10f \n' % p2)
print('exact = %.10f \n' %exact)
print('\n')
