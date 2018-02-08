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


class FaceTrainer():
    def __init__(self, hog_parameters, args, silent=False):
        self.silent = silent

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

        self.source_dir = args.source_dir
        self.hog_p = hog_parameters
        self.annotations = self.load_wiki_annotations(args.annotations)
        self.id = -1
        self.anno = None

        self.samples = list()
        self.neg_samples = list()

        if not self.silent:
            cv2.namedWindow('wnd', cv2.WINDOW_NORMAL)
            cv2.namedWindow('hog', cv2.WINDOW_NORMAL)

    def close(self):
        try:
            cv2.destroyAllWindows()
        except SystemError:
            pass

    def load_wiki_annotations(self, anno_path):
        # The format is for  WIKI faces dataset only https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
        return sio.loadmat(anno_path)['wiki'][0]

    def get_wiki_annotation(self, id):
        dob = self.annotations['dob'][0][0]
        photo_taken = self.annotations['photo_taken'][0][0]
        full_path = self.annotations['full_path'][0][0]
        gender = self.annotations['gender'][0][0]
        name = self.annotations['name'][0][0]
        face_location = self.annotations['face_location'][0][0]
        face_score = self.annotations['face_score'][0][0]
        second_face_score = self.annotations['second_face_score'][0][0]

        return WikiAnnotation(
            index = id,
            dob = dob[id], # date of birth (Matlab serial date number)
            ptaken = photo_taken[id], # year when the photo was taken
            path = full_path[id], # path to file
            gender = gender[id], # 0 for female and 1 for male, NaN if unknown
            name = name[id], # name of the celebrity
            loc = face_location[id], # location of the face. To crop the face in Matlab run
                                                    # img(face_location(2):face_location(4),face_location(1):face_location(3),:))
            score = face_score[id], # detector score (the higher the better). Inf implies that no face was found in the image and the face_location then just returns the entire image
            s2 = second_face_score[id], # detector score of the face with the second highest score. This is useful to ignore images with more than one face. second_face_score is NaN if no second face was detected.
            cname = 0,
            cid = 0,
            #cname = annotations['wiki'][0][0][8][id], # list of all celebrity names. IMDB only
            #cid = annotations['wiki'][0][0][9][id], # index of celebrity name. IMDB only
        )

    def next_image(self, silent = False):
        self.id += 1
        self.anno = self.get_wiki_annotation(self.id)

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

    def predict(self):
        svm = self.svm
        svec = svm.getSupportVectors()[0]
        rho = -svm.getDecisionFunction(0)[0]
        svec = np.append(svec, rho)
        print(svec.dtype, svec.shape)
        self.hog.setSVMDetector(svec)
        (rects, weights) = self.hog.detectMultiScale(self.img_src,
            winStride=(8, 8), padding=(0, 0), scale=1.1, useMeanshiftGrouping=False)
        predictions = np.column_stack([weights, rects])
        print('predictions', predictions)
        predictions = predictions[np.lexsort(np.fliplr(predictions).T)]
        print('predictions', predictions)
        #numpy.sort
        self.draw_predict(predictions[-3:])

    def draw_predict(self, predictions):
        cv2.namedWindow('wnd', cv2.WINDOW_NORMAL)

        src_size = self.img_src.shape
        img = np.repeat(self.img_src[:, :, np.newaxis], 3, axis=2)

        for predict in predictions:
            p1 = (int(predict[1]), int(predict[2]))
            p2 = (int(predict[3]), int(predict[4]))
            cv2.rectangle(img, p1, p2, (0, 255, 255), 1)

        cv2.resizeWindow('wnd', src_size[1]*2, src_size[0]*2)
        cv2.imshow('wnd', img)

    def hog_compute(img, p):
        width, height, depth = img.shape
        
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
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size,
            nbins, deriv_aperture, win_sigma, histogram_norm_type, threshold, gamma_correction, nlevels)
        hog.save("hog.xml")

        win_stride = p.w_stride # Not affecting length of output
        padding = p.padding
        hist = hog.compute(img, win_stride, padding) # Omit locations to compute whole image
        return hist

    def compute(self):
        i_size = np.array((width, height))
        win_size = np.array(hog_p.w) # Decrease length of output
        win_stride = np.array(hog_p.w_stride) # Not affecting length of output
        padding = np.array(hog_p.padding)
        scan_size = (i_size + padding * 2 - win_size) // win_stride * win_stride + win_size
        win_dim = np.array(calc_num_of_blocks(scan_size, win_size, win_stride))

        hist = hog_compute(img_src, hog_p)

        cursor = [0, 0]
        draw(img_src, hog_p, 'wnd', cursor, hist, anno)

    def draw(self, background, hog_p, wnd, cursor, hist, anno):
        img = np.copy(background)
        width, height, depth = img.shape

        for loc in anno.loc:
            cv2.rectangle(img, (int(loc[0]), int(loc[1])), (int(loc[2]), int(loc[3])), (0, 255, 255), 1)

        i_size = np.array((width, height))
        win_size = np.array(hog_p.w) # Decrease length of output
        win_stride = np.array(hog_p.w_stride) # Not affecting length of output
        padding = np.array(hog_p.padding)

        block_size = np.array(hog_p.b) # In pixels
        block_stride = np.array(hog_p.b_stride) # In pixels
        cell_size = np.array(hog_p.c) # In pixels
        nbins = np.array(hog_p.nbins)

        #print('i_size', i_size)
        scan_size = (i_size + padding * 2 - win_size) // win_stride * win_stride + win_size
        #print('scan_size', scan_size)
        win_dim = np.array(calc_num_of_blocks(scan_size, win_size, win_stride))
        #print('win_dim', win_dim)
        block_dim = np.array(calc_num_of_blocks(win_size, block_size, block_stride))
        #print('block_dim', block_dim)
        cell_dim = block_size // cell_size
        #print('cells in block', cell_dim, cell_dim[0]*cell_dim[1])
        #print('num of hist', hist.shape[0]/nbins)
        #print('num of hist.', win_dim[0]*block_dim[0]*cell_dim[0]*win_dim[1]*block_dim[1]*cell_dim[1])
        shape = win_dim
        shape = np.append(shape, block_dim)
        shape = np.append(shape, cell_dim)
        shape = np.append(shape, [nbins,])
        #print('num of bins', hist.shape[0])
        #print('num of calc. bins', np.prod(shape))
        #print('shape', shape)
        hist = np.reshape(np.array(hist), shape)
        #print(hist)
        #return
        win = hist[cursor[0]][cursor[1]]
        #print(win.shape)
        #print(win[0][0])
        #print(win[0][1])

        p1 = np.array(cursor * win_stride - padding)
        p2 = p1 + win_size
        cdim = win_size // cell_size

        len = cell_size // 2 - [2, 2]
        
        c1 = np.array(p1)
        for y in range(cdim[1]):
            for x in range(cdim[0]):
                c2 = tuple(np.add(c1, hog_p.c))
                cv2.rectangle(img, tuple(c1), c2, (255, 0, 255), 1)
                c = (c1 + c2) // 2
                cv2.line(img, tuple(c - len), tuple(c + len), (0, 255, 0), 1)
                c1[0] += hog_p.c[0]
            c1[0] = p1[0]
            c1[1] += hog_p.c[1]
        cv2.rectangle(img, tuple(p1), tuple(p2), (0, 255, 0), 1)
        
        width, height, depth = img.shape
        cv2.resizeWindow(wnd, height*2, width*2)
        cv2.imshow(wnd, img)

    def train(self):
        batch_size = 500

        for i in range(batch_size):
            ret = self.next_image()
            while not ret:
                ret = self.next_image()

        # Convert objects to Numpy Objects
        samples = np.float32(self.samples)
        labels = np.zeros((batch_size,), dtype=np.int32)
        labels[:] = 1
        print('positive samples', samples.shape)

        # Generate random negative samples
        #neg_data = np.random.random((batch_size*4,)+samples[0].shape).astype('float32')
        neg_samples = np.float32(self.neg_samples)
        neg_labels = np.zeros((batch_size*4,), dtype=np.int32)
        print('negative samples', neg_samples.shape)

        samples = np.concatenate([samples, neg_samples])
        labels = np.concatenate([labels, neg_labels])

        '''entropy = np.zeros((batch_size*2,), dtype=np.float32)
        for i in range(batch_size*2):
            entropy[i] = scipy.stats.entropy(samples[i])
        print(entropy)
        print(labels)'''

        # Shuffle Samples
        rand = np.random.RandomState(321)
        shuffle = rand.permutation(len(samples))
        samples = samples[shuffle]
        labels = labels[shuffle]

        print(samples.shape)
        print(labels.shape)

        # Create SVM classifier
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC) # cv2.ml.SVM_C_SVC, cv2.ml.ONE_CLASS
        self.svm.setKernel(cv2.ml.SVM_RBF) # cv2.ml.SVM_LINEAR, SVM::INTER, cv2.ml.SVM_RBF
        # svm.setDegree(0.0)
        self.svm.setGamma(5.383)
        # svm.setCoef0(0.0)
        self.svm.setC(2.67)
        # svm.setNu(0.0)
        # svm.setP(0.0)
        # svm.setClassWeights(None)

        # Train
        self.svm.train(samples, cv2.ml.ROW_SAMPLE, labels)
        #self.svm.save('svm_data.dat')

        '''td = cv2.TrainData.create(InputArray samples, int layout, InputArray responses, InputArray varIdx=noArray(), InputArray sampleIdx=noArray(), InputArray sampleWeights=noArray(), InputArray varType=noArray())
        err = svm.calcError(samples, cv2.ml.ROW_SAMPLE, labels)
        print(err)'''


def main():
    #pydoc.writedoc("cv2.HOGDescriptor")

    hog_p = HogParameters(
        w = (128, 128), # Window size, in pixels. default [64,128]
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
    print('norm=0')
    ft = FaceTrainer(hog_p, ARGS, silent=True)

    ft.train()

    ret = ft.next_image()
    while not ret:
        ret = ft.next_image()
    self.predict()

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