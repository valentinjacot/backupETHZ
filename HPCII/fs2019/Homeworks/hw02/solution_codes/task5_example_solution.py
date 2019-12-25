### HPCSE II Spring 2019
### HW 2 - Task 4: 1-D Laplacian Approximation

###############################################################################
### Import Modules
###############################################################################
import numpy as np
import matplotlib.pyplot as plt



###############################################################################
### Subtask 5. Visualisation
###############################################################################

mu = -2.0
gamma = 3.0

var = 0.5*gamma**2

# Lambda functions for Cauchy distribution
lk_cauchy  = lambda x, m, g: (1/np.math.pi*g/(g**2+(x-m)**2))

# Lambda functions for Gaussian likelihood
lk_laplace  = lambda x, m, v: lk_cauchy(mu, mu, gamma) * np.exp(-0.5 * (x-m)**2 / v)


x = np.linspace(-7, 3, 100)

cy = list(map(lambda d: lk_cauchy(d, mu, gamma), x))
gy = list(map(lambda d: lk_laplace(d, mu, var), x))

plt.plot(x, cy, label = 'Cauchy distribution')
plt.plot(x, gy, label = 'Unnormalized Laplace approximation')
plt.xlabel("x")
plt.ylabel("$pdf(x)$")
plt.legend(loc='lower center')
plt.show()



###############################################################################
### END
###############################################################################
