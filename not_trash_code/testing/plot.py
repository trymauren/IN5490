import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,20,10000)

def y(x):
	f = 1
	phase_shift = 0
	A = 1
	theta = A*np.sin(f*x+phase_shift)
	return theta*40


plt.plot(x,y(x))
plt.show()