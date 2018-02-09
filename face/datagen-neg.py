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
from uuid import uuid4

from datagen import FaceGenerator

ARGS = None


class NegativeGenerator(FaceGenerator):
    def spawn(self, img, preview):
        # Use opencv HAAR detector to avoid faces and generate negative samples
        # https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html

        w_size = [64, 96]
        depth = 1
        if len(img.shape)==3:
            height, width, depth = img.shape
        else:
            height, width = img.shape
        if w_size[0] > width or w_size[1] > height:
            return

        if self.face_cascade is None: self.face_cascade = cv2.CascadeClassifier('./pretrained/haarcascades/haarcascade_frontalcatface.xml')
        faces = self.face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=3)

        if len(faces)==0:
            sys.stdout.write('^')
            sys.stdout.flush()
            stride_x = [64, 0]
            stride_y = [0, 64]

            p1 = np.array([0, 0])
            p2 = p1 + w_size
            while True:
                while True:
                    n_sample = np.copy(img[p1[1]:p2[1], p1[0]:p2[0]])
                    faces = self.face_cascade.detectMultiScale(n_sample, scaleFactor=1.05, minNeighbors=3)
                    cv2.imwrite(os.path.join(self.negative_dir, str(self.fseq).zfill(6)+'.jpg'), n_sample)
                    self.fseq += 1
                    p1 = p1 + stride_x
                    p2 = p2 + stride_x
                    if p2[0] >= width:
                        p1[0] = 0
                        p2 = p1 + w_size
                        break
                p1 = p1 + stride_y
                p2 = p2 + stride_y
                if p2[1] >= height:
                    break
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

    def preview(self, img, annotations, wname):
        img = self.gray2rgb(img)

        faces = annotations[0]
        eyeslist = annotations[1]
        i = 0
        for (x, y, w, h) in faces:
            if w and h:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 128, 128), 1)
                eyes = eyeslist[i]
                if len(eyes)==2:
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0,255,0), 1)
            i += 1

        self.draw(img, wname)
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
    with NegativeGenerator(ARGS) as dg:
        dg.gen(preview=ARGS.preview)


if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Generate negative samples with supplied directory of source images""")
    parser.add_argument(
        '--source_dir',
        type=str,
        default='../../data/coco/',
        help='Path to the source data.'
    )
    parser.add_argument(
        '--dest_dir',
        type=str,
        default='../../data/face/processed',
        help='Path to the processed data.'
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Display samples in window for preview.'
    )
    ARGS, unknown = parser.parse_known_args()
    if unknown:
        print(unknown)
        raise Exception('Unknown argument')
    main()