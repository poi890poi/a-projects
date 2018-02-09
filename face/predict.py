import pydoc
import cv2
import math
import numpy as np
import scipy.io as sio
import scipy.stats
from scipy.misc import imresize
from skimage.feature import hog
from skimage import data, exposure

import argparse
import collections
import os.path
import sys

from datagen import DataGenerator, ImageProcessor, ImageProcessorParameters
from train import FaceTrainer

ARGS = None

HogParameters = collections.namedtuple('HogParameters', 'w, b, b_stride, c, nbins, aperture, sigma, norm, t, g, nlevels, w_stride, padding')
WikiAnnotation = collections.namedtuple('WikiAnnotation', 'index, dob, ptaken, path, gender, name, loc, score, s2, cname, cid')

class FaceDetector(DataGenerator):
    def __init__(self, args):
        self.args = args
        super(FaceDetector, self).__init__(args)

    def set_detector(self, detector):
        self.detector = detector
        
    def spawn(self, img, preview):
        print('spawn')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        channels = cv2.split(gray)
        gray = channels[0]

        predictions = self.detector.predict(gray)
        for face in predictions:
            p1 = (int(face[1]), int(face[2]))
            p2 = (int(face[3]), int(face[4]))
            cv2.rectangle(img, p1, p2, (0, 255, 255), 1)
        print('predictions', predictions)

        if preview:
            self.draw(gray, 'gray')
            self.draw(img, 'img')
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

    hog_p = HogParameters(
        w = (64, 96), # Window size, in pixels. default [64,128]
        b = (16, 16), # Block size, in pixels. default [16,16]
        b_stride = (8, 8), # Block stride, in pixels. default [8,8]
        c = (8, 8), # Cell size, in pixels. default [8,8]
        nbins = 9, # Number of bins. default 9
        aperture = 1, # aperture_size Size of the extended Sobel kernel, must be 1, 3, 5 or 7. default 1
        sigma = -1, # Windows sigma. Gaussian smoothing window parameter. default -1
        norm = 0, # Histogram normalization method. default 'L2Hys'
        t = 0.2, # L2 Hysterisis threshold. normalization method shrinkage. default 0.2
        g = True, # Flag to specify whether the gamma correction preprocessing is required or not. default true
        nlevels = 64, # Maximum number of detection window increases. default 64
        w_stride = (8, 8), # Window stride, in pixels
        padding = (8, 8), # Padding of source image for window (not for block nor cell)
    )
    ft = FaceTrainer(hog_p, ARGS, silent=True)
    ft.load('svm_data.dat')

    with FaceDetector(ARGS) as fd:
        fd.set_detector(ft)
        fd.gen(preview=True, imp=ImageProcessorParameters(
            convert_gray = ImageProcessor.GRAY_NONE,
        ))

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Clean up images and transform to generate more samples""")
    parser.add_argument(
        '--source_dir',
        type=str,
        default='../../data/face/wiki-face/extracted/wiki',
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