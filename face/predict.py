import pydoc
import cv2
import math
import numpy as np
import scipy.io as sio
import scipy.stats
from scipy.misc import imresize
from scipy.sparse.csgraph import connected_components
from skimage import data, exposure, feature, measure, filters
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV

import argparse
import collections
import os
import sys
import pickle

import linecache, tracemalloc
import psutil
import objgraph

from utilities import DataUtilities, ImageUtilities, DirectoryWalker, ViewportManager
from config import HyperParam

ARGS = None

def memory_usage_psutil(checkpoint):
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    print()
    print(checkpoint, mem)
    return mem

def display_top(snapshot, key_type='lineno', limit=10):
    print()
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

def main():
    #pydoc.writedoc("cv2.HOGDescriptor")
    if ARGS is None:
        profiling = True
        debug = False
        source_dir = '../../data/face/wiki-face/extracted/wiki'
    else:
        profiling = False
        debug = ARGS.debug
        source_dir = ARGS.source_dir

    with open('svm.dat', 'rb') as f:
        svm = pickle.load(f)

    while True:
        f = DirectoryWalker().get_a_file(directory=source_dir, filters=['.jpg'])
        if f is None:
            print('No more positive sample T_T')
            break

        stride = 1
        scale = 1.2

        cell_size = np.array(HyperParam.cell_size)
        block_size = np.array(HyperParam.cell_size) * np.array(HyperParam.block_size)
        #window_size = (96, 64)
        #descriptor_shape = (11, 7, 2, 2, 9)
        window_size = np.array(HyperParam.window_size)
        window_stride = np.array(HyperParam.window_stride)
        descriptor_shape = tuple(((window_size-block_size)/cell_size + np.array([1, 1])).astype(dtype=np.int)) + tuple(HyperParam.block_size) + (HyperParam.nbins,)
        descriptor_len = int(np.prod(descriptor_shape))

        source = cv2.imread(f.path, 1)
        src_shape = np.array(source.shape, dtype=np.float32)
        gray = ImageUtilities.preprocess(source, convert_gray=cv2.COLOR_RGB2YCrCb, maxsize=640)
        gray = ImageUtilities.cell_resize(gray, cell_size=cell_size*stride)
        print('source shape', gray.shape)
        
        height, width, *rest = gray.shape

        heat_map = np.zeros(gray.shape, dtype=np.uint8)

        window_count = 0
        positive_count = 0
        for i in range(8):
            if height < window_size[0] or width < window_size[1]:
                print('sample too small')
                continue

            print('pyramid', i, width, height)
            mrate = [src_shape[0]/height, src_shape[1]/width]
            #print('mrate', i, mrate)
            resized = imresize(gray, [height, width])

            #src_size = np.array([width, height])
            #window_count = np.prod(((src_size - window_size)/cell_size + np.array([1, 1])).astype(dtype=np.int))
            #print('windows:', window_count)

            hog_image = None
            if debug:
                hog, hog_image = feature.hog(resized, orientations=HyperParam.nbins, pixels_per_cell=cell_size, cells_per_block=HyperParam.block_size, block_norm='L2-Hys',
                    visualise=True, transform_sqrt=True, feature_vector=False)
                hog_image = ImageUtilities.gray2rgb((hog_image*255).astype(dtype=np.uint8))
            else:
                hog = feature.hog(resized, orientations=HyperParam.nbins, pixels_per_cell=cell_size, cells_per_block=HyperParam.block_size, block_norm='L2-Hys',
                    visualise=False, transform_sqrt=True, feature_vector=False)

            if debug: preview = ImageUtilities.gray2rgb(resized)
            #print(descriptor_shape[:2])
            slen = DataUtilities.get_strider_len(matrix=hog, window=descriptor_shape[:2])//(stride*stride)
            if slen <= 0: continue
            window_count += slen
            hoglist = np.zeros((slen,)+(descriptor_len,), dtype=np.float32)
            poslist = np.zeros((slen, 2), dtype=np.uint16)
            for index, pos, subarray in DataUtilities.strider(matrix=hog, window=descriptor_shape[:2], stride=stride):
                #print(index, pos)
                hoglist[index, :] = np.reshape(subarray, (1, descriptor_len))
                poslist[index, :] = pos

            predictions = svm.predict(hoglist)
            #print('predictions', predictions.shape)
            i = 0
            for p in predictions:
                if p==1:
                    pos = poslist[i]
                    p1 = (pos[1]*cell_size[1], pos[0]*cell_size[0])
                    p2 = (pos[1]*cell_size[1]+window_size[1]-1, pos[0]*cell_size[0]+window_size[0]-1)
                    if debug: cv2.rectangle(preview, p1, p2, (0, 255, 0), 1)
                    positive_count += 1

                    #yslice = np.array([pos[0]*cell_size[0], pos[0]*cell_size[0]+window_size[0]])*mrate[0].astype(np.uint16)
                    #xslice = np.array([pos[1]*cell_size[1], pos[1]*cell_size[1]+window_size[1]])*mrate[1].astype(np.uint16)
                    #heat_map[yslice[0]:yslice[1], xslice[0]:xslice[1]] += 1
                    heat_map[p1[1]:p2[1], p1[0]:p2[0]] += 1
                    #hnm = img[p1[1]:p2[1], p1[0]:p2[0], :]
                i += 1

            if debug:
                canvas = ViewportManager().open('preview', shape=preview.shape, blocks=(2, 2))

                # Process heat_map and get bounding boxes
                binary = cv2.threshold(heat_map, 0, 255, cv2.THRESH_BINARY)[1]
                labels = measure.label(binary)
                #print(labels, blobs_labels)
                for region in measure.regionprops(labels):
                    #p1 = np.array([region.bbox[0]*mrate[1], region.bbox[1]*mrate[0]]).astype(dtype=np.int)
                    #p2 = np.array([region.bbox[2]*mrate[1], region.bbox[3]*mrate[0]]).astype(dtype=np.int)
                    p1 = np.array([region.bbox[1], region.bbox[0]]).astype(dtype=np.int)
                    p2 = np.array([region.bbox[3], region.bbox[2]]).astype(dtype=np.int)
                    print(region.area*np.prod(mrate), region.bbox, p1, p2)
                    cv2.rectangle(preview, tuple(p1), tuple(p2), (0, 0, 255), 1)
                labels = imresize(labels, [height, width])
                labels = ImageUtilities.gray2rgb(labels)
                ViewportManager().put('preview', labels, (1,1))

                ViewportManager().put('preview', preview, (0,0))
                ViewportManager().put('preview', hog_image, (0,1))

                resized = imresize(heat_map, [height, width])

                resized = ImageUtilities.gray2rgb(resized*32)
                ViewportManager().put('preview', resized, (1,0))
                
                ViewportManager().update('preview')
            
                k = ViewportManager().wait_key()
                if k in (ViewportManager.KEY_ENTER, ViewportManager.KEY_SPACE):
                    pass
                elif k in (ViewportManager.KEY_RIGHT,):
                    # Next image
                    break
                elif k in (ViewportManager.KEY_DOWN,):
                    # Save in hard-negative-mining dataset
                    break
                elif k in (ViewportManager.KEY_DOWN,):
                    # Save in positive dataset
                    break

            height = int(height/scale//cell_size[0]*cell_size[0])
            width = int(width/scale//cell_size[1]*cell_size[1])
            heat_map = imresize(heat_map, [height, width])

        # End of a image (pyramid)
        if width <= 0  or height <= 0: continue
        print('window_count', positive_count, '/', window_count, (window_count-positive_count))

        canvas = ViewportManager().open('preview', shape=source.shape, blocks=(1, 1))
        mrate = [src_shape[0]/height, src_shape[1]/width]

        # Process heat_map and get bounding boxes
        binary = cv2.threshold(heat_map, 0, 255, cv2.THRESH_BINARY)[1]
        labels = measure.label(binary)
        for region in measure.regionprops(labels):
            p = region.area/(region.bbox[3]-region.bbox[1])/(region.bbox[2]-region.bbox[0])
            print('region', region.bbox_area, region.area, region.filled_area, region.solidity)
            p1 = np.array([region.bbox[1]*mrate[1], region.bbox[0]*mrate[0]]).astype(dtype=np.int)
            p2 = np.array([region.bbox[3]*mrate[1], region.bbox[2]*mrate[0]]).astype(dtype=np.int)
            cv2.rectangle(source, tuple(p1), tuple(p2), (0, 255, 0), 1)
        ViewportManager().put('preview', source, (0, 0))
        ViewportManager().update('preview')

        k = ViewportManager().wait_key()
        if k in (ViewportManager.KEY_ENTER, ViewportManager.KEY_SPACE):
            pass

        if profiling:
            objgraph.show_refs([svm, heat_map, hoglist], filename='fd-graph.png')

            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot)

            print()
            break

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Clean up images and transform to generate more samples""")
    parser.add_argument(
        '--source_dir',
        type=str,
        #default='../../data/coco',
        default='../../data/face/wiki-face/extracted/wiki',
        #default='../../data/face/processed/positive',
        help='Path to the data.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Process debug information. (slower)'
    )
    parser.add_argument(
        '--annotations',
        type=str,
        default='../../data/face/wiki-face/extracted/wiki/wiki.mat',
        help='Path to annotations.'
    )
    ARGS, unknown = parser.parse_known_args()
    if unknown:
        print(unknown)
        raise Exception('Unknown argument')
    main()