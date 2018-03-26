import numpy as np

def fer(args):
    with open('../../data/face/fer2013/fer2013', 'r') as f:
        count = 10
        for line in f:
            if len(line) > 2000:
                sample = line.split(',')
                img = np.array(sample[1].split()).astype(dtype=np.uint8).reshape((48, 48))
                #img = np.array(sample[1].split())
                #print(sample[1])
                #img = np.fromstring(sample[1]).astype(dtype=np.uint8)
                print(sample[0], img, sample[2])
                count -= 1
                if count<= 0: break
