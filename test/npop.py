import numpy as np

a = np.array([1, 2, 3, 4], dtype=np.uint8)
print('a[1:2]', a[1:2])
print('a[1:8]', a[1:8])
print('a[2::]', a[2::])
