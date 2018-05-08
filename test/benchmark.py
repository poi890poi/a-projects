import time
import numpy as np
import timeit

import cv2
import jpeg4py as jpeg

print('BytesIO write', timeit.timeit(
stmt='''b.write(b"B")''',
setup='''from io import BytesIO
b = BytesIO()''', number=100)*50000000)

print('bytes type concat', timeit.timeit(
stmt='''b += b"B"''',
setup='''b = b""''', number=100)*50000000)

with open('sample.jpg', 'rb') as f:
    bindata = np.frombuffer(f.read(), np.uint8)

    t_ = time.perf_counter() * 1000
    for i in range(10):
        img = cv2.imdecode(bindata, 1)
    t_ = time.perf_counter()*1000 - t_
    print('imdecode', t_)

    t_ = time.perf_counter() * 1000
    for i in range(10):
        img = jpeg.JPEG(bindata).decode()
    t_ = time.perf_counter()*1000 - t_
    print('jpeg4py', t_)

print('np.zeros', timeit.timeit(stmt='npar = np.zeros((1920, 1080, 3), dtype=np.uint8)', setup='import numpy as np', number=1000)*10000)

setup = '''
import numpy as np
a = np.zeros((640, 480, 3), dtype=np.uint8)
'''

print('astype float', timeit.timeit(
stmt='a = a.astype(dtype=np.float)',
setup=setup, number=100)*500)

print('astype float32', timeit.timeit(
stmt='a = a.astype(dtype=np.float32)',
setup=setup, number=100)*500)

setup = '''
import numpy as np
a = np.ones((640, 480, 3), dtype=np.float)
'''

print('float div', timeit.timeit(
stmt='a = a / 255.',
setup=setup, number=100)*500)

setup = '''
import numpy as np
a = np.ones((640, 480, 3), dtype=np.float32)
'''

print('float32 div', timeit.timeit(
stmt='a = a / 255.',
setup=setup, number=100)*500)

l = [1] * 1000000

t_ = time.time() * 1000
for i in range(1000000):
    l[i] /= 2.
t_ = time.time()*1000 - t_
print('loop', t_)

t_ = time.time() * 1000
l = list(map(lambda x: x/2, l))
t_ = time.time()*1000 - t_
print('map', t_)

t_ = time.time() * 1000
for i in range(1000000):
    s = 'one' + str(1.)
t_ = time.time()*1000 - t_
print('concat', t_)

t_ = time.time() * 1000
for i in range(1000000):
    s = '{}{}'.format('one', 1.)
t_ = time.time()*1000 - t_
print('format', t_)
