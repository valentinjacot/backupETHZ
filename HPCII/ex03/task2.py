import numpy as np
import matplotlib.pyplot as plt
def f(x, mu, var):
	return (2*np.math.pi*var**2)**(-0.5) * np.exp(-(x-mu)**2/ (2*var**2))
def g(x, lam):
	return lam * np.exp(-lam*x)
def h(x):
	if x > 4.5:
		return 1
	else:
		return 0
N=10000
data =  np.random.normal(0, 1,N)
est3 = 0
for i in range(N):
	est3 += h(data[i])
print ("Î(3)= ")
print(est3/N) 
print(" \n")
data =  np.random.exponential(1,N)
est4 = 0
for i in range(N):
	est3 += (h(data[i])*f(data[i], 0,1)) / g(data[i],1)

print ("Î(4)= ") 
print(est4/N) 
print(" \n")

print ("F(4.5)= ") 
exact = 1-0.5*(1+np.math.erf(4.5/np.math.sqrt(2)))
print(exact)
	
