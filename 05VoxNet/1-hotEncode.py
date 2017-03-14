import numpy as np

a = np.arange(10)
b = np.zeros((10, 11))
b[np.arange(10), a] = 1
label = b[2][:-1]
print(label)