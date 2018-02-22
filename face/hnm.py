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
import sys, hashlib
import pickle

from datagen import DataGenerator, ImageProcessor
from utilities import DataUtilities, ImageUtilities
from train import FaceTrainer
from config import HyperParam

ARGS = None

WikiAnnotation = collections.namedtuple('WikiAnnotation', 'index, dob, ptaken, path, gender, name, loc, score, s2, cname, cid')

class FaceDetector(DataGenerator):
    def __init__(self, args):
        self.args = args
        super(FaceDetector, self).__init__(args)

    def init(self):
        with open('svm.dat', 'rb') as f:
            self.svm = pickle.load(f)

        self.face_cascade = cv2.CascadeClassifier('./pretrained/haarcascades/haarcascade_frontalcatface.xml')

        DataUtilities.prepare_dir(self.args.dest_dir, empty=True)

        self.hnm_index = 0
        self.limit = 2000
        self.skip = 10000
        
    def spawn(self, img, preview):
        img = self.cell_resize(img)
        print('spawn', img.shape)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        channels = cv2.split(gray)
        gray = channels[0]

        height, width, depth = img.shape

        print('img shape', img.shape)

        faces = self.face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=3)
        if len(faces):
            print('pass')
            return

        if preview:
            hist, hog_image = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys',
                visualise=True, transform_sqrt=True, feature_vector=False)
        else:
            hist = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys',
                visualise=False, transform_sqrt=True, feature_vector=False)
            
        if preview: hog_image = (hog_image*255).astype(dtype=np.uint8)
        #hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 255))

        window_size = HyperParam.window_size
        descriptor_size = (int((window_size[0]-16)/8+1), int((window_size[1]-16)/8+1), 2, 2, 9)

        feature_vector_size = np.prod(descriptor_size)
        print('feature_vector_size', feature_vector_size)
        img_hog_size = hist.shape
        print('image hog', img_hog_size)

        wnd_count = (img_hog_size[0]-descriptor_size[0]+1)*(img_hog_size[1]-descriptor_size[1]+1)
        print('wnd_count', wnd_count)
        hog_arr = np.zeros((wnd_count,)+(feature_vector_size,), dtype=np.float32)
        coords = np.zeros((wnd_count, 2), dtype=np.uint16)
        print('hog_arr', hog_arr.shape)

        i = 0
        scan = [0, 0]
        while True:
            while True:
                wnd_hog = hist[scan[0]:scan[0]+descriptor_size[0], scan[1]:scan[1]+descriptor_size[1], :, :, :]
                hog_arr[i, :] = np.reshape(wnd_hog, (1, feature_vector_size))
                coords[i, :] = scan
                #p = self.svm.predict(np.reshape(wnd_hog, (1, feature_vector_size)))
                i += 1
                scan[1] += 1
                if scan[1]+descriptor_size[1] >= img_hog_size[1]:
                    scan[1] = 0
                    break
            scan[0] += 1
            if scan[0]+descriptor_size[0] >= img_hog_size[0]:
                break

        if preview:
            canvas = np.zeros((height, width*2, depth), dtype=np.uint8)
            canvas[:,0:width,:] = np.repeat(gray[:, :, np.newaxis], 3, axis=2)
            canvas[:,width:width*2,:] = np.repeat(hog_image[:, :, np.newaxis], 3, axis=2)

        predictions = self.svm.predict(hog_arr)
        i = 0
        c_scan = [0, 0]
        for p in predictions:
            if p==1:
                c = coords[i]
                #print(p, c)
                p1 = (c[1]*8, c[0]*8)
                p2 = (c[1]*8+window_size[1], c[0]*8+window_size[0])
                hnm = img[p1[1]:p2[1], p1[0]:p2[0], :]

                if preview:
                    cv2.rectangle(canvas, p1, p2, (0, 255, 0), 1)
                    canvas[c_scan[0]*window_size[0]:(c_scan[0]+1)*window_size[0], width+c_scan[1]*window_size[1]:width+(c_scan[1]+1)*window_size[1], :] = hnm

                h = hashlib.new('ripemd160')
                h.update(hnm.tobytes())
                fname = h.hexdigest()
                #outpath = os.path.join(self.args.dest_dir, str(self.hnm_index).zfill(5)+'.jpg')
                outpath = os.path.join(self.args.dest_dir, fname+'.jpg')
                cv2.imwrite(outpath, hnm)
                self.hnm_index += 1
                sys.stdout.write('.')
                sys.stdout.flush()
                
                if self.hnm_index==self.limit:
                    print()
                    print('sample count limit reached :<')
                    sys.exit()

                c_scan[1] += 1
                if (c_scan[1]+1)*64 >= width:
                    c_scan[1] = 0
                    c_scan[0] += 1
                if (c_scan[0]+1)*8 >= height:
                    break
            i += 1

        if preview:
            self.draw(canvas, 'canvas')
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
        fd.init()
        fd.gen(preview=False)

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Clean up images and transform to generate more samples""")
    parser.add_argument(
        '--source_dir',
        type=str,
        default='../../data/coco',
        #default='../../data/face/wiki-face/extracted/wiki',
        ##default='../../data/face/processed/positive',
        help='Path to the data.'
    )
    parser.add_argument(
        '--dest_dir',
        type=str,
        default='../../data/face/processed/hnm-tmp',
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