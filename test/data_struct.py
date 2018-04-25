import copy

d = {'one': 1, 'two': 2}
print(d)
c = dict(d)
print(c)
c['two'] = 3
print(d, c)

d = {'d': d}
c = dict(d)
c['d']['two'] = 3
print('new dict', d, c)

d = {'d': {'one': 1, 'two': 2}}
c = copy.deepcopy(d)
c['d']['two'] = 3
print('deepcopy', d, c)
