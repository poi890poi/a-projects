import pydoc
import cv2
import math
import numpy as np
from scipy.misc import imresize


import argparse
import collections
import os.path
import sys, hashlib
from uuid import uuid4
import os, stat


class DataUtilities():
    @staticmethod
    def hash_str(text):
        h = hashlib.new('ripemd160')
        h.update(text.encode('utf-8'))
        return h.hexdigest()

    @staticmethod
    def prepare_dir(head, trail='', create=True, empty=False):
        directory = DataUtilities.norm_join_path(head, trail)
        if empty and os.path.isdir(directory):
            print()
            print('Removing directory:', directory)
            tmp = DataUtilities.norm_join_path(head, hash_str(directory))
            os.chmod(directory, stat.S_IWRITE)
            os.rename(directory, tmp)
            os.removedirs(tmp)
        if create and not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    @staticmethod
    def norm_join_path(head, trail):
        return os.path.normpath(os.path.join(head, trail))

    @staticmethod
    def get_strider_len(matrix, window):
        height, width, *rest = matrix.shape
        return int((height-window[0]+1)*(width-window[1]+1))

    @staticmethod
    def strider(matrix, window, stride=1):
        height, width, *rest = matrix.shape
        i = 0
        scan = [0, 0]
        while True:
            while True:
                yield (i, scan, matrix[scan[0]:scan[0]+window[0], scan[1]:scan[1]+window[1], :, :, :])
                i += 1
                scan[1] += stride
                if scan[1]+window[1] >= width:
                    scan[1] = 0
                    break
            scan[0] += stride
            if scan[0]+window[0] >= height:
                break


class ImageUtilities():
    @staticmethod
    def cell_resize(img, cell_size=(8, 8)):
        depth = 1
        if len(img.shape)==3:
            height, width, depth = img.shape
        else:
            height, width = img.shape

        height = height//cell_size[0]*cell_size[0]
        width = width//cell_size[1]*cell_size[1]
        
        if depth==1:
            return np.copy(img[0:height, 0:width])

        return np.copy(img[0:height, 0:width, :])

    @staticmethod
    def preprocess(img, convert_gray=None, maxsize=512):
        size = np.array(img.shape)
        r = 1.
        if size[0] > maxsize or size[1] > maxsize:
            r = min(maxsize/size[0], maxsize/size[1])
        size = ((size.astype('float32'))*r).astype('int16')
        img = imresize(img, size)
        
        if convert_gray is not None:
            # Convert to YCrCb and keep only Y channel.
            img = cv2.cvtColor(img, convert_gray)
            channels = cv2.split(img)
            img = channels[0]

        depth = 1
        if len(img.shape)==3:
            height, width, depth = img.shape
        else:
            height, width = img.shape
        
        # Denoise and equalize. Note 2018-02-09: Denoising benefits HAAR face detector significantly
        if depth==1:
            img = cv2.fastNlMeansDenoising(img)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img  = clahe.apply(img)
        elif depth==3:
            img = cv2.fastNlMeansDenoisingColored(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            channels = cv2.split(img)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img[:, :, 0] = clahe.apply(channels[0])
            img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        else:
            raise TypeError('ImageProcessor::preprocess() expects image of 1 or 3 channels')

        return img

    @staticmethod
    def gray2rgb(gray):
        # Clone single channel to RGB
        return np.repeat(gray[:, :, np.newaxis], 3, axis=2)


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class DirectoryWalker(metaclass=Singleton):
    def __init__(self):
        self.walkers = dict()

    def get_walker_create(self, directory):
        if directory not in self.walkers:
            walker = {
                'directory': directory,
                'iterator': self.recursive_file_iterator(directory),
                'pointer': 0,
                'history': list(),
                'filter': list(),
            }
            self.walkers[directory] = walker
        return self.walkers[directory]

    def recursive_file_iterator(self, directory):
        for entry in os.scandir(directory):
            if entry.is_dir():
                yield from self.recursive_file_iterator(entry.path)
            else:
                yield entry

    def get_a_file(self, directory, filters=None):
        walker = self.get_walker_create(directory)
        for f in walker['iterator']:
            if filters:
                ext = os.path.splitext(f.name)[1].lower()
                if ext not in filters:
                    continue
            return f
        return None

class ArrayStrider():
                
    def run(self):
        pass

    def window_callback(self):
        raise NotImplementedError('ImageStrider::window_callback() is not implmented')


class ViewportManager(metaclass=Singleton):
    KEY_ENTER = 0
    KEY_SPACE = 1
    KEY_UP = 10
    KEY_DOWN = 11
    KEY_LEFT = 12
    KEY_RIGHT = 13

    def __init__(self):
        self.viewports = dict()

    def open(self, wname, shape, blocks=(1, 1)):
        shape = np.array(shape)
        if len(shape) < 3: shape = (shape,)+(3,)
        #print('ViewportManager::open()', shape)
        shape[0] *= blocks[0]
        shape[1] *= blocks[1]
        if wname not in self.viewports:
            cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
        self.viewports[wname] = {
            'canvas' : np.zeros(shape, dtype=np.uint8)
        }
        cv2.resizeWindow(wname, shape[1], shape[0])
        return self.viewports[wname]['canvas']

    def put(self, wname, img, block):
        canvas = self.viewports[wname]['canvas']
        height, width, *rest = img.shape
        canvas[0+block[0]*height:height+block[0]*height, 0+block[1]*width:width+block[1]*width, :] = img[:,:,:]

    def update(self, wname):
        canvas = self.viewports[wname]['canvas']
        cv2.imshow(wname, canvas)

    def wait_key(self):
        while True:
            k = cv2.waitKeyEx(0)
            if k in (1, 27, -1): # esc
                print()
                print('User terminated :>')
                sys.exit()
            elif k in (13,):
                return self.KEY_ENTER
            elif k in (32,):
                return self.KEY_SPACE 
            elif k in (2490368, ):
                return self.KEY_UP
            elif k in (2621440, ):
                return self.KEY_DOWN
            elif k in (2424832,):
                return self.KEY_LEFT
            elif k in (2555904,):
                return self.KEY_RIGHT
            else:
                print('unregistered key_code:', k)
                pass
            sleep(12)
        return -1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for prefix in self.viewports:
            cv2.destroyWindow(prefix)

