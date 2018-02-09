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

from datagen import ImageProcessor, RecursiveDirectoryWalkerManager

ARGS = None

HogParameters = collections.namedtuple('HogParameters', 'w, b, b_stride, c, nbins, aperture, sigma, norm, t, g, nlevels, w_stride, padding')

class FaceTrainer(ImageProcessor):
    def __init__(self, hog_parameters, args, silent=False):
        self.silent = silent

        super(FaceTrainer, self).__init__()

        p = hog_parameters
        win_size = p.w # Decrease length of output
        block_size = p.b # In pixels
        block_stride = p.b_stride # In pixels
        cell_size = p.c # In pixels
        nbins = p.nbins
        deriv_aperture = p.aperture
        win_sigma = p.sigma
        histogram_norm_type = p.norm
        threshold = p.t
        gamma_correction = p.g
        nlevels = p.nlevels
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size,
            nbins, deriv_aperture, win_sigma, histogram_norm_type, threshold, gamma_correction, nlevels)
        self.hog.save("hog.xml")
        print('descriptor size', self.hog.getDescriptorSize())

        self.hog_p = hog_parameters
        self.id = -1
        self.anno = None

    def close(self):
        try:
            cv2.destroyAllWindows()
        except SystemError:
            pass

    def load(self, path):
        self.svm = cv2.ml.SVM_load(path)

        (rho, alpha, svi) = self.svm.getDecisionFunction(0)
        svec = self.svm.getSupportVectors().ravel()
        print('svec', svec.dtype, svec.shape)
        svec = np.append(svec, -rho)
        print('load', svec.dtype, svec.shape)
        print('size', self.hog.getDescriptorSize())
        self.hog.setSVMDetector(svec)
        return

        svec = self.svm.getSupportVectors()[0]
        rho = -self.svm.getDecisionFunction(0)[0]
        svec = np.append(svec, rho)
        print('load pretrained svm', svec.dtype, svec.shape)
        self.hog.setSVMDetector(svec)

    def predict(self, img):
        (rects, weights) = self.hog.detectMultiScale(img,
            winStride=(8, 8), padding=(0, 0), scale=1.05, useMeanshiftGrouping=False)
        predictions = np.column_stack([weights, rects])
        predictions = predictions[np.lexsort(np.fliplr(predictions).T)]
        return predictions
        #self.draw_predict(predictions[-3:])

    def train(self, positive_dir, negative_dir):
        batch_size = 2000
        increment = batch_size//10
        
        dir_walk_mgr = RecursiveDirectoryWalkerManager()

        # Get positive samples
        i = batch_size
        samples = None
        p_len = 0
        while i:
            if i%increment==0:
                sys.stdout.write('.')
                sys.stdout.flush()
            f = dir_walk_mgr.get_a_file(directory=positive_dir, filters=['.jpg'])
            if f is None:
                print('Not enough positive samples T_T')
                break
            img = cv2.imread(os.path.normpath(f.path), 1) # Load as RGB for compatibility
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            channels = cv2.split(img)
            img = channels[0]

            hist = self.hog.compute(img, winStride=(0, 0), padding=(0, 0))
            if samples is None:
                samples = np.zeros((batch_size,)+hist.shape, dtype=np.float32)
            samples[p_len,:] = hist
            p_len += 1
            i -= 1

        print(samples.shape)
        positive_samples = np.copy(samples[0:p_len, :])
        print(samples.shape)

        # Get negative samples
        samples = None
        n_len = p_len  * 4
        i = n_len
        pt = 0
        while i:
            if i%increment==0:
                sys.stdout.write('.')
                sys.stdout.flush()
            f = dir_walk_mgr.get_a_file(directory=negative_dir, filters=['.jpg'])
            if f is None:
                print('Not enough negative samples T_T')
                break
            img = cv2.imread(os.path.normpath(f.path), 1) # Load as RGB for compatibility
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            channels = cv2.split(img)
            img = channels[0]

            hist = self.hog.compute(img, winStride=(0, 0), padding=(0, 0))
            if samples is None:
                samples = np.zeros((n_len,)+hist.shape, dtype=np.float32)
            try:
                samples[pt,:] = hist
            except:
                pass
            pt += 1
            i -= 1

        samples = np.concatenate([positive_samples, samples])
        print(samples.shape)

        # Convert to numpy array of float32 and create labels
        print(p_len, n_len)
        labels = np.zeros((p_len+n_len,), dtype=np.int32)
        labels[0:p_len] = 1

        # Shuffle Samples
        rand = np.random.RandomState(321)
        shuffle = rand.permutation(len(samples))
        samples = samples[shuffle]
        labels = labels[shuffle]

        print(samples.shape)
        print(labels.shape)
        print('Training...')

        # Create SVM classifier
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC) # cv2.ml.SVM_C_SVC, cv2.ml.ONE_CLASS
        self.svm.setKernel(cv2.ml.SVM_LINEAR) # cv2.ml.SVM_LINEAR, SVM::INTER, cv2.ml.SVM_RBF
        # svm.setDegree(0.0)
        self.svm.setGamma(5.383)
        # svm.setCoef0(0.0)
        self.svm.setC(2.67)
        # svm.setNu(0.0)
        # svm.setP(0.0)
        # svm.setClassWeights(None)

        # Train
        self.svm.train(samples, cv2.ml.ROW_SAMPLE, labels)
        self.svm.save('svm_data.dat')

        '''td = cv2.TrainData.create(InputArray samples, int layout, InputArray responses, InputArray varIdx=noArray(), InputArray sampleIdx=noArray(), InputArray sampleWeights=noArray(), InputArray varType=noArray())
        err = svm.calcError(samples, cv2.ml.ROW_SAMPLE, labels)
        print(err)'''


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

    positive_dir = os.path.join(ARGS.train_dir, 'positive')
    negative_dir = os.path.join(ARGS.train_dir, 'negative')
    ft.train(positive_dir, negative_dir)

    while True:
        k = cv2.waitKeyEx(0)
        if k in (1, 27, -1): # esc
            print('break')
            break
        elif k in (13, 32): # enter, space
            ret = ft.next_image()
            while not ret:
                ret = ft.next_image()
            ft.predict()
        elif k in (2490368, ): # up
            if cursor[1] > 0:
                cursor[1] -= 1
            draw(img_src, hog_p, 'wnd', cursor, hist, anno)
        elif k in (2621440, ): # down
            if cursor[1] < win_dim[1] - 1:
                cursor[1] += 1
            draw(img_src, hog_p, 'wnd', cursor, hist, anno)
        elif k in (2424832,): # left
            if cursor[0] > 0:
                cursor[0] -= 1
            draw(img_src, hog_p, 'wnd', cursor, hist, anno)
        elif k in (2555904,): # right
            if cursor[0] < win_dim[0] - 1:
                cursor[0] += 1
            draw(img_src, hog_p, 'wnd', cursor, hist, anno)
        else:
            print('unregistered key_code:', k)

    ft.close()

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Clean up images and transform to generate more samples""")
    parser.add_argument(
        '--train_dir',
        type=str,
        default='../../data/face/processed',
        help='Path to train data.'
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        default='../../data/face/wiki',
        help='Path to test data.'
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