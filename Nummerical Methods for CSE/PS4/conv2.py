import numpy as np
import math
e=[1,0.8]
le=[1,1]

for i in range(1,20):
	e+=[e[i]+math.sqrt(e[i-1])]
	le+=[math.log(e[i])]
	print(le[i]/le[i-1])

