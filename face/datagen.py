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

from getdata import recursive_file_iterator

ARGS = None

HogParameters = collections.namedtuple('HogParameters', 'w, b, b_stride, c, nbins, aperture, sigma, norm, t, g, nlevels, w_stride, padding')
WikiAnnotation = collections.namedtuple('WikiAnnotation', 'index, dob, ptaken, path, gender, name, loc, score, s2, cname, cid')

def calc_num_of_blocks(map_size, block_size, block_stride):
    return [(map_size[0]-block_size[0])//block_stride[0]+1, (map_size[1]-block_size[1])//block_stride[1]+1]


def load_image(path, wnd):
    img = cv2.imread(path)
    #img = cv2.resize(img, (64, 64))
    width, height, depth = img.shape

    cv2.resizeWindow(wnd, width, height)

    return img


def next_file(refite):
    for f in refite:
        print(f)
        return load_image(f, 'wnd')


class DataGenerator():
    def __init__(self, args):
        self.args = args

    def gen(self):
        pass

    def finalize(self):
        pass

    def next_image(self):
        self.id += 1

        # Load image from file
        path = os.path.normpath(os.path.join(self.source_dir, self.anno.path[0]))
        img = cv2.imread(path, 1)
        if img is None: return 0 # Broken image
        
        # Convert to YCrCb and keep only Y channel
        cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        channels = cv2.split(img)
        self.img_src = channels[0]
        src_size = self.img_src.shape
        # Equalize
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.img_src  = clahe.apply(self.img_src )
        # Convert back to RGB for visualization
        img = np.repeat(self.img_src[:, :, np.newaxis], 3, axis=2)

        # Process faces
        try:
            cv2.destroyAllWindows('face')
        except SystemError:
            pass
        fi = 0
        for loc in self.anno.loc:
            p1 = (int(loc[0]), int(loc[1]))
            p2 = (int(loc[2]), int(loc[3]))
            cv2.rectangle(img, p1, p2, (0, 255, 255), 1)

            face = np.copy(self.img_src[int(loc[1]):int(loc[3]), int(loc[0]):int(loc[2])])
            face_size = np.array(face.shape)
            if face_size[0] < 16 or face_size[1] < 16:
                # Invalid sample
                return 0

            # Resize cropped faces
            s_size = self.hog_p.w
            face = imresize(face, s_size)

            hist, hog_image = hog(face, orientations=self.hog_p.nbins, pixels_per_cell=self.hog_p.c, cells_per_block=(1, 1), visualise=True)
            hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            hist = np.reshape(hist, hist.shape + (1,))
            # Display face
            if not self.silent:
                print('sk hist', hist.shape)
                print('hist', hist[-16:])
                print('hog_image', hog_image.shape)
                wname = 'face::'+str(fi)
                cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(wname, 512, 512)
                cv2.imshow(wname, face)
                wname = 'face::hog::'+str(fi)
                cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(wname, 512, 512)
                cv2.imshow(wname, hog_image)

            # Generate positive samples
            hist = self.hog.compute(face, padding=(0, 0), locations=[[0,0],])
            #hist = self.hog.compute(face, winStride=(4, 4), padding=(0, 0))
            if not self.silent:
                print('cv hist', hist.shape)
                print('hist', hist[-16:])
            self.samples.append(hist)

            # TODO: Flip to create one additional positive sample
            
            fi += 1

        # Generate negative samples
        # TODO: Randomly crop soure image and flip to generate negative samples
        stride_x = [8, 0]
        stride_y = [0, 8]
        w_size = np.array(self.hog_p.w)
        i_size = self.img_src.shape
        p1 = np.array([0, 0])
        p2 = p1 + w_size
        img = np.repeat(self.img_src[:, :, np.newaxis], 3, axis=2)
        while True:
            while True:
                #print(p1, p2)
                for loc in self.anno.loc:
                    #print(p1[0]-loc[2], loc[1]-p2[0], p1[1]-loc[3], loc[1]-p2[1])
                    if loc[2] <= p1[0] or loc[1] >= p2[0] or loc[3] <= p1[1] or loc[1] >= p2[1]:
                        #face = np.copy(self.img_src[int(loc[1]):int(loc[3]), int(loc[0]):int(loc[2])])
                        n_sample = np.copy(self.img_src[p1[1]:p2[1], p1[0]:p2[0]])
                        hist = self.hog.compute(n_sample, padding=(0, 0), locations=[[0,0],])
                        self.neg_samples.append(hist)
                p1 = p1 + stride_x
                p2 = p2 + stride_x
                if p2[0] >= i_size[1]:
                    p1[0] = 0
                    p2 = p1 + w_size
                    break
            p1 = p1 + stride_y
            p2 = p2 + stride_y
            if p2[1] >= i_size[0]:
                break
        if not self.silent:
            cv2.imshow('wnd', img)

        print(self.anno.name)

        if not self.silent:
            cv2.resizeWindow('wnd', src_size[1]*2, src_size[0]*2)
            cv2.imshow('wnd', img)
            hist, hog_image = hog(self.img_src, orientations=self.hog_p.nbins, pixels_per_cell=self.hog_p.c, cells_per_block=(1, 1), visualise=True)
            hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            src_size = hog_image.shape
            cv2.resizeWindow('hog', src_size[1]*2, src_size[0]*2)
            cv2.imshow('hog', hog_image)
        return 1

def main():
    datagen = DataGenerator(ARGS)

    datagen.gen()

    ret = ft.next_image()
    while not ret:
        ret = ft.next_image()

    datagen.finalize()

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
    parser.add_argument(
        '--train_dir',
        type=str,
        default='../../data/face/train/',
        help='Path to training data.'
    )
    ARGS, unknown = parser.parse_known_args()
    if unknown:
        print(unknown)
        raise Exception('Unknown argument')
    main()