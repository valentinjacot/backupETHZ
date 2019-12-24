import math
import numpy as np
def fp(x,h):
	temp=(2*x+h)/2
	return (2*math.cos(temp)*math.sin(h/2))/h
def f1(x,h):
	return (math.sin(x+h)-math.sin(x))/h
print (" f prime of sin for x= 1.2")
space=np.logspace(-20,0,20)
exact_value = math.cos(1.2)
print (exact_value)
for h  in space:
	print ('f w/o cancellation:',fp(1.2,h), " \t error : ", fp(1.2,h)-exact_value, '\n f with cancellation', f1(1.2,h), '\t error: ', f1(1.2,h)-exact_value)
