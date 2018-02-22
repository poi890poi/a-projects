import pydoc
import cv2
import math
import numpy as np
import scipy.io as sio
import scipy.stats
from scipy.misc import imresize
from skimage import data, exposure, feature

import argparse
import collections
import os.path
import sys
import pickle

from utilities import DataUtilities, ImageUtilities, DirectoryWalker, ViewportManager
from config import HyperParam, Model

ARGS = None


def main():
    #pydoc.writedoc("cv2.HOGDescriptor")
    with open('svm.dat', 'rb') as f:
        svm = pickle.load(f)

    pp = 0
    nn = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    cell_size = HyperParam.cell_size
    window_size = HyperParam.window_size
    descriptor_shape = (int((window_size[0]-16)/8+1), int((window_size[1]-16)/8+1), 2, 2, 9)
    descriptor_len = int(np.prod(descriptor_shape))

    while True:
        f = DirectoryWalker().get_a_file(directory=ARGS.positive_dir, filters=['.jpg'])
        if f is None: break

        source = cv2.imread(f.path, 1)
        gray = ImageUtilities.preprocess(source, convert_gray=cv2.COLOR_RGB2YCrCb, maxsize=640)
        gray = imresize(gray, window_size)
        hog = feature.hog(gray, orientations=9, pixels_per_cell=cell_size, cells_per_block=(2, 2), block_norm='L2-Hys',
            visualise=False, transform_sqrt=True, feature_vector=True)
        p = svm.predict(np.reshape(hog, (1, descriptor_len)))

        pp += 1
        if p[0]==1:
            # True positive
            tp += 1
            sys.stdout.write('o')
        else:
            # False negative
            fn += 1
            sys.stdout.write('x')
        sys.stdout.flush()

    while True:
        f = DirectoryWalker().get_a_file(directory=ARGS.negative_dir, filters=['.jpg'])
        if f is None: break

        source = cv2.imread(f.path, 1)
        gray = ImageUtilities.preprocess(source, convert_gray=cv2.COLOR_RGB2YCrCb, maxsize=640)
        gray = imresize(gray, window_size)
        hog = feature.hog(gray, orientations=9, pixels_per_cell=cell_size, cells_per_block=(2, 2), block_norm='L2-Hys',
            visualise=False, transform_sqrt=True, feature_vector=True)
        p = svm.predict(np.reshape(hog, (1, descriptor_len)))

        nn += 1
        if p[0]==0:
            # True negative
            tn += 1
            sys.stdout.write('.')
        else:
            # False positive
            fp += 1
            sys.stdout.write('#')
        sys.stdout.flush()

    print()
    print('total samples:', pp+nn)
    print('positive samples:', pp)
    print('negative samples:', nn)
    print('true positive:', tp)
    print('true negative:', tn)
    print('false positive:', fp)
    print('false negative:', fn)
    print('correct:', (tp+tn)/(pp+nn))
    print('detection rate:', tp/pp)
    print('error:', (fp+fn)/(pp+nn))

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Clean up images and transform to generate more samples""")
    parser.add_argument(
        '--positive_dir',
        type=str,
        default='../../data/face/val/positive',
    )
    parser.add_argument(
        '--negative_dir',
        type=str,
        default='../../data/face/val/negative',
    )
    ARGS, unknown = parser.parse_known_args()
    if unknown:
        print(unknown)
        raise Exception('Unknown argument')
    main()