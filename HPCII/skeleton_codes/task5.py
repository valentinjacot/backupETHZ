import numpy as np
import matplotlib.pyplot as plt


def cauchy(x, x0, gamma):
    return 1/np.math.pi * (gamma/ ((x-x0)*(x-x0)+gamma*gamma))
def gaussian(x, mu, var):
	return (2*np.math.pi*var**2)**(-0.5) * np.exp(-(x-mu)**2/ (2*var**2))
def laplace(x, x0, gamma):
	return (2*np.math.pi*gamma**2)**(-0.5) * np.exp(-(x-x0)**2/ (gamma**2))

data =  np.linspace(-10, 10, 100)
cauchy_val= cauchy(data, -2, 1)
laplace_val = laplace(data, -2, 1)

plt.subplot(121)
plt.plot(data, cauchy_val, 'b^-')
plt.title('Cauchy distribution')
plt.ylabel('values')
plt.xlabel('data')
plt.subplot(122)
plt.plot(data, laplace_val, 'b^-')
plt.title('Laplace approximation')
plt.xlabel('data')
plt.show()  
