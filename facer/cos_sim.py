import scipy
import numpy as np
import sklearn.metrics, sklearn.preprocessing
import scipy.spatial, scipy.cluster

for _ in range(64):
    samples = np.random.rand(1024, 128)

    target = np.random.rand(1, 128)
    target = sklearn.preprocessing.normalize(target)
    samples = sklearn.preprocessing.normalize(samples)

    print()

    tree = scipy.spatial.KDTree(samples)
    print(tree.query(target[0]))

    d_min = 9999
    i_ = -1
    for i, v_ in enumerate(samples):
        d_ = scipy.spatial.distance.cosine(target[0], v_)
        if d_ < d_min:
            d_min = d_
            i_ = i

    print('min cosine similarity', i_, d_min)

"""
clusters = scipy.cluster.vq.kmeans(samples, 2)
for v_ in clusters[0]:
    v_ = scipy.cluster.vq.whiten(v_)
    d_ = scipy.spatial.distance.cosine(target, v_)
    print(d_)
"""

"""
s_ = sklearn.metrics.pairwise.cosine_similarity(samples, samples)
print(s_)

print()
for v_ in samples:
    d_ = scipy.spatial.distance.cosine(target, v_)
    print(d_)
"""

#print(scipy.__version__)
#print(samples, target)