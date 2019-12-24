import numpy as np
import math
def F(x):
	return x*math.exp(x)-1

def secant(x0,x1,F,atol,rtol):
	fo=F(x0)
	for i in 100:
		fn=F(x1)
		s=fn*(x1-x0)/(fn-f0)
		x0=x1
		x1=x1-s
		if abs(s) < max(atol, rtol*min(abs(x0),abs(x1)):
			return x1
		f0=fn

res= secant(0,5,F,1e-8,1e-6)
print(res)
