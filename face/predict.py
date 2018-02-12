import pydoc
import cv2
import math
import numpy as np
import scipy.io as sio
import scipy.stats
from scipy.misc import imresize
from skimage import data, exposure, feature
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV

import argparse
import collections
import os.path
import sys
import pickle

from datagen import DataGenerator, ImageProcessor
from train import FaceTrainer

ARGS = None

HogParameters = collections.namedtuple('HogParameters', 'w, b, b_stride, c, nbins, aperture, sigma, norm, t, g, nlevels, w_stride, padding')
WikiAnnotation = collections.namedtuple('WikiAnnotation', 'index, dob, ptaken, path, gender, name, loc, score, s2, cname, cid')

class FaceDetector(DataGenerator):
    def __init__(self, args):
        self.args = args
        super(FaceDetector, self).__init__(args)

    def load_svm(self):
        with open('svm.dat', 'rb') as f:
            self.svm = pickle.load(f)
        
    def spawn(self, img, preview):
        print('spawn')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        channels = cv2.split(gray)
        gray = channels[0]

        height, width, depth = img.shape

        print('img shape', img.shape)

        hist, hog_image = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys',
            visualise=True, transform_sqrt=True, feature_vector=False)
        hog_image = (hog_image*255).astype(dtype=np.uint8)
        #hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 255))

        preview = np.zeros((height, width*2, depth), dtype=np.uint8)
        preview[:,0:width,:] = np.repeat(gray[:, :, np.newaxis], 3, axis=2)
        preview[:,width:width*2,:] = np.repeat(hog_image[:, :, np.newaxis], 3, axis=2)

        p = self.svm.predict(np.reshape(hist, (1, 2772)))
        print(p)

        self.draw(preview, 'preview')
        self.wait_user() # Always wait for user input if preview is displayed

    def wait_user(self):
        while True:
            k = cv2.waitKeyEx(0)
            if k in (1, 27, -1): # esc
                print()
                print('User terminated :>')
                sys.exit()
            elif k in (13, 32): # enter, space
                self.gen(preview=True)
            elif k in (2490368, ): # up
                pass
            elif k in (2621440, ): # down
                pass
            elif k in (2424832,): # left
                pass
            elif k in (2555904,): # right
                pass
            else:
                print('unregistered key_code:', k)

def main():
    #pydoc.writedoc("cv2.HOGDescriptor")

    with FaceDetector(ARGS) as fd:
        fd.load_svm()
        fd.gen(preview=True)

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Clean up images and transform to generate more samples""")
    parser.add_argument(
        '--source_dir',
        type=str,
        default='../../data/coco',
        #default='../../data/face/wiki-face/extracted/wiki',
        #default='../../data/face/processed/positive',
        help='Path to the data.'
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