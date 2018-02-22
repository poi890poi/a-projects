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

from datagen import ImageProcessor, RecursiveDirectoryWalkerManager
from config import HyperParam, Model

ARGS = None

HogParameters = collections.namedtuple('HogParameters', 'w, b, b_stride, c, nbins, aperture, sigma, norm, t, g, nlevels, w_stride, padding')

class FaceTrainer(ImageProcessor):
    def __init__(self, args, silent=False):
        self.silent = silent

        super(FaceTrainer, self).__init__()

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

        #pydoc.writedoc("cv2.ml.SVM_create")

    def predict(self, img):
        (rects, weights) = self.hog.detectMultiScale(img,
            winStride=(4, 4), padding=(0, 0), scale=1.03, useMeanshiftGrouping=False)
        predictions = np.column_stack([weights, rects])
        predictions = predictions[np.lexsort(np.fliplr(predictions).T)]
        return predictions
        #self.draw_predict(predictions[-3:])

    def compute_hog(self, img):
        height, width, *rest = img.shape
        t_size = HyperParam.window_size

        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        channels = cv2.split(img)
        img = channels[0]
        img = imresize(img, t_size)
        return (feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys',
            visualise=False, transform_sqrt=True, feature_vector=True), img)

    def train(self, positive_dir, negative_dir, hnm_dir):
        model = Model.svc()

        batch_size = 6000
        increment = batch_size//100
        
        dir_walk_mgr = RecursiveDirectoryWalkerManager()

        # Get positive samples
        print('Loading positive samples...')
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

            hist, img = self.compute_hog(img)
            if samples is None:
                samples = np.zeros((batch_size,)+hist.shape, dtype=np.float32)
            samples[p_len,:] = hist
            p_len += 1
            i -= 1

        print(samples.shape)
        positive_samples = np.copy(samples[0:p_len, :])
        print('Positive samples loaded:', positive_samples.shape)

        # Get negative samples
        print('Loading negative samples...')
        samples = None
        n_len = p_len  * 10
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

            hist, img = self.compute_hog(img)
            if samples is None:
                samples = np.zeros((n_len,)+hist.shape, dtype=np.float32)
            try:
                samples[pt,:] = hist.ravel()
            except:
                pass
            pt += 1
            i -= 1

        print('Negative samples loaded:', samples.shape)
        samples = np.concatenate([positive_samples, samples])

        # Get hard-negative-mining samples
        for di in range(10):
            directory = os.path.normpath(os.path.join(hnm_dir, str(di+1).zfill(4)))
            if not os.path.isdir(directory):
                break
            print('Loading hard-negative-mining samples...', directory)

            hnm_samples = None
            t_len = batch_size * 10
            i = t_len
            print('target sample size', i)
            pt = 0
            while i:
                if i%increment==0:
                    sys.stdout.write('.')
                    sys.stdout.flush()

                f = dir_walk_mgr.get_a_file(directory=directory, filters=['.jpg'])
                if f is None:
                    print('Not enough hard-negative-mining samples T_T')
                    break

                img = cv2.imread(os.path.normpath(f.path), 1) # Load as RGB for compatibility
                if img is None:
                    continue

                hist, img = self.compute_hog(img)
                if hnm_samples is None:
                    hnm_samples = np.zeros((t_len,)+hist.shape, dtype=np.float32)
                try:
                    hnm_samples[pt,:] = hist.ravel()
                except:
                    pass
                pt += 1
                i -= 1

            hnm_samples = np.copy(hnm_samples[0:pt, :])
            print('HNM samples loaded:', hnm_samples.shape)
            samples = np.concatenate([samples, hnm_samples])
        
        print('Total samples:', samples.shape)

        # Convert to numpy array of float32 and create labels
        labels = np.zeros((samples.shape[0],), dtype=np.int32)
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
        model.fit(samples, labels)
        print(model.best_score_)
        with open('svm.dat', 'wb') as f:
            pickle.dump(model, f)

        '''td = cv2.TrainData.create(InputArray samples, int layout, InputArray responses, InputArray varIdx=noArray(), InputArray sampleIdx=noArray(), InputArray sampleWeights=noArray(), InputArray varType=noArray())
        err = svm.calcError(samples, cv2.ml.ROW_SAMPLE, labels)
        print(err)'''


def main():
    #pydoc.writedoc("cv2.HOGDescriptor")

    ft = FaceTrainer(ARGS, silent=True)

    positive_dir = os.path.join(ARGS.train_dir, 'positive')
    negative_dir = os.path.join(ARGS.train_dir, 'negative')
    hnm_dir = os.path.join(ARGS.train_dir, 'hnm')
    ft.train(positive_dir, negative_dir, hnm_dir)

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