import numpy as np

a = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
print('a[1:2]', a[1:2])
print('a[1:8]', a[1:8])
print('a[2::]', a[2::])

a = a.tolist()
print()
print('list:')
print('a[1:2]', a[1:2])
print('a[1:8]', a[1:8])
print('a[2::]', a[2::])

from sklearn.utils import shuffle

b = ['one', 'two', 'three', 'four', 'five']
print()
print(a, b)
a, b = shuffle(a, b)
print(a, b)
b = b + ['six']
print(a, b)
a = a[2::]
b = b[2:2+len(a)]
print(a, b)