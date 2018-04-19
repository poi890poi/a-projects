import numpy as np
import cv2
import random
import time

img = np.zeros((480, 640, 3), dtype=np.uint8)

clusters = [
    {
        'count': 64,
        'position': (60, 60, 120, 120, 40),
        'color': (255, 0, 255),
    },
    {
        'count': 64,
        'position': (240, 240, 100, 100, 40),
        'color': (0, 255, 255),
    },
    {
        'count': 64,
        'position': (320, 60, 80, 80, 40),
        'color': (255, 255, 0),
    },
    {
        'count': 64,
        'position': (280, 100, 80, 80, 40),
        'color': (255, 255, 0),
    },
]

rectangles = []

font = cv2.FONT_HERSHEY_SIMPLEX
r_index = 0
for c in clusters:
    for i in range(c['count']):
        x = c['position'][0] + random.randrange(c['position'][4])
        y = c['position'][1] + random.randrange(c['position'][4])
        w = c['position'][2] + random.randrange(c['position'][4])
        h = c['position'][3] + random.randrange(c['position'][4])
        rectangles.append([x, y, w, h])
        #cv2.rectangle(img, (x, y), (x+w, y+h), c['color'])
        #cv2.putText(img, str(r_index).zfill(3), (x, y-10), font, 0.7, c['color'], 1, cv2.LINE_AA)
        r_index += 1

def get_iou(r1, r2):
    x1 = max(r1[0], r2[0])
    x2 = min(r1[0]+r1[2], r2[0]+r2[2])
    if x2 < x1:
        return 0
    y1 = max(r1[1], r2[1])
    y2 = min(r1[1]+r1[3], r2[1]+r2[3])
    if y2 < y1:
        return 0
    aoi = (x2-x1)*(y2-y1)
    return aoi / (r1[2]*r1[3] + r2[2]*r2[3] - aoi)

def get_min_iou(iou_table, cluster, candidate):
    if len(cluster)==0:
        return 0
    iou = []
    for i, r_index in enumerate(cluster):
        iou.append(iou_table[r_index][candidate])
    return np.amin(np.array(iou))

def group_rectangles_miniou(rectangles, threshold=0.4):
    lr = len(rectangles)
    iou_table = np.zeros((lr, lr), dtype=np.float)
    for i1, r1 in enumerate(rectangles):
        for i2, r2 in enumerate(rectangles):
            if i2 <= i1: continue
            iou = get_iou(r1, r2)
            iou_table[i1][i2] = iou
    for i1, r1 in enumerate(rectangles):
        for i2, r2 in enumerate(rectangles):
            if i2 < i1:
                iou_table[i1][i2] = iou_table[i2][i1]

    #print(iou_table)
    clusters = [[0],]
    for ir, r in enumerate(rectangles):
        #print()
        #print('check rectangle...', ir)
        new_cluster = True
        for ic, c in enumerate(clusters):
            if ir in c: # The rectangle is already in the cluster
                new_cluster = False
                break

            iou = np.zeros((len(c),), dtype=np.float)
            for r_index, c_member in enumerate(c):
                iou[r_index] = iou_table[c_member][ir]
            min_iou = np.amin(np.array(iou))
            
            #print(c, ir, min_iou)
            if min_iou > threshold:
                clusters[ic].append(ir)
                new_cluster = False
                #print('add to cluster', clusters)
                break
        
        # The rectangle is not in any cluster
        if new_cluster:
            clusters.append([ir,])
            #print('new_cluster', clusters)
    
    return clusters

t_ = time.time()
for i in range(100):
    group_rectangles_miniou(rectangles)
print((time.time()-t_)*1000/100)

cv2.imshow('viewport', img)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
