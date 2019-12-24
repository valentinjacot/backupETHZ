import numpy as np
import matplotlib.pyplot as plt


def f1(x):
	return (x+1)**4 - (x-1)**4 +1
def f2(x):
	return -x**3 + 8*x +1
def f3(x):
	return (-11/3)*(x**3) + 8*x**2 +(11/3)

t1=np.arange(-1,0,0.01)
t2=np.arange(0,1,0.01)
t3=np.arange(1,2,0.01)
plt.plot(t1,f1(t1))
plt.plot(t2,f2(t2))
plt.plot(t3,f3(t3))
plt.grid(True)
plt.show()
