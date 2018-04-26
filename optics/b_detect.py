import cv2
import numpy as np
import random

count = 16

shape = (count, 256, 256, 1)
height = shape[1]
width = shape[2]
depth = shape[3]
images = np.zeros(shape, dtype=np.uint8) + np.random.poisson(16, shape)

for i in range(count//2):
    spots = np.zeros(shape[1:4], dtype=np.uint8)
    for s in range((i+2)*2):
        y = random.randint(0, height-1)
        x = random.randint(0, width-1)
        for d in range(depth):
            spots[y, x, d] = random.randint(64, 192)
    spots = cv2.GaussianBlur(spots, (3, 3), 0).reshape(shape[1:4])
    images[i] = images[i] + spots

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

for i, img in enumerate(images):
    cv2.imwrite('../../data/optics/{}.bmp'.format(str(i).zfill(2)), img)
    mean = np.mean(img)
    var = np.var(img)
    #print('mean: {}, var: {}'.format(mean, var))

    # create a CLAHE object (Arguments are optional).
    if depth==3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[0]
    else:
        lab = img.reshape(height, width).astype(np.uint8)
    #print(lab.shape, lab.dtype)
    #lab = clahe.apply(lab)
    edges = cv2.Canny(lab, 100, 200)
    cv2.imwrite('../../data/optics/{}-edge.bmp'.format(str(i).zfill(2)), edges)
    mean = np.mean(edges)
    var = np.var(edges)
    print('edge, mean: {}, var: {}'.format(mean, var))

    print()    

