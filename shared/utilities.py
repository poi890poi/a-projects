import pydoc
import cv2
import math
import numpy as np
from scipy.misc import imresize

import tensorflow as tf

import argparse
import collections
import os.path
import sys, hashlib
from uuid import uuid4
import os, stat, shutil

import random

def hash_str(text):
    h = hashlib.new('ripemd160')
    h.update(text.encode('utf-8'))
    return h.hexdigest()

class DataUtilities():

    """def handleRemoveReadonly(func, path, exc):
    excvalue = exc[1]
    if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
        func(path)
    else:
        raise

    shutil.rmtree(filename, ignore_errors=False, onerror=handleRemoveReadonly)"""

    @staticmethod
    def prepare_dir(head, trail='', create=True, empty=False):
        directory = DataUtilities.norm_join_path(head, trail)
        if empty and os.path.isdir(directory):
            print()
            print('Removing directory:', directory)
            tmp = DataUtilities.norm_join_path(head, hash_str(directory))
            #os.chmod(directory, stat.S_IWRITE)
            os.rename(directory, tmp)
            #os.chmod(tmp, stat.S_IRWXU|stat.S_IRWXG|stat.S_IRWXO)
            shutil.rmtree(tmp)
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
    def save_dataset(data, labels, dir, epoch):
        i = 0
        for img in data:

            i += 1

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
    def rect_fit_points(points, target_ar=1., expand=2.):
        x = np.min(points[:, 0:1])
        y = np.min(points[:, 1:2])
        w = np.max(points[:, 0:1]) - x
        h = np.max(points[:, 1:2]) - y

        scaling = [1., 1.]
        s_ = target_ar * h / w
        if s_ > 1:
            scaling[0] = s_
        else:
            scaling[1] = 1. / s_
        w_ = (w * scaling[0] * expand - w) / 2.
        h_ = (h * scaling[1] * expand - h) / 2.
        x -= w_
        y -= h_
        w += w_ * 2.
        h += h_ * 2.

        return (x, y, w, h)

    @staticmethod
    def fit_resize(img, maxsize=(320, 200), interpolation=cv2.INTER_NEAREST):
        height, width, *_ = img.shape
        scaling = min(maxsize[1]/height, maxsize[0]/width, 1.)

        # scipy.misc.imresize changes range [min, max] of pixel values and MUST be avoided when pixel array is used as input for CNN
        #img = imresize(img, (np.array([height, width], dtype=np.float)*scaling).astype(dtype=np.int), interp='bilinear')
        img = cv2.resize(img, tuple((np.array([width, height], dtype=np.float)*scaling).astype(dtype=np.int)), interpolation=interpolation)

        return (img, scaling)

    @staticmethod
    def rect_fit_ar(rect, bound, target_ar, mrate=1.0, crop=False):
        # Default behavior is to contain original rectangle. Specify crop=True to crop instead.
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
        ar = w/h
        if ar >= target_ar: # Original rect too wide
            if not crop:
                h_new = w/target_ar
                h_new *= mrate
                w_new = w * mrate
            else:
                w_new = h
                h_new = h
        else: # Original rect too tall
            if not crop:
                w_new = h*target_ar
                w_new *= mrate
                h_new = h * mrate
            else:
                w_new = w
                h_new = w
        h_diff = (h - h_new)
        y += int(h_diff/2)
        w_diff = (w - w_new)
        x += int(w_diff/2)
        h = int(h_new)
        w = int(w_new)
        if x < bound[0] or y < bound[1] or w > bound[2] or h > bound[3]:
            x = rect[0]
            y = rect[1] - rect[2]//4
            w = rect[2]
            h = rect[3] + rect[2]//2
            return (x, y, w, h)
        return (x, y, w, h)

    @staticmethod
    def rect_to_bb(rect, mrate=None):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y

        if mrate is not None:
            x = int(x/mrate[1])
            y = int(y/mrate[0])
            w = int(w/mrate[1])
            h = int(h/mrate[0])
    
        return (x, y, w, h)

    @staticmethod
    def is_color(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        channels = cv2.split(img)
        return np.mean(channels[1])

    @staticmethod
    def preprocess(img, convert_gray=None, equalize=True, denoise=True, maxsize=512):
        size = np.array(img.shape)
        r = 1.
        if maxsize > 0 and (size[0] > maxsize or size[1] > maxsize):
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
        if equalize or denoise:
            if depth==1:
                if denoise: img = cv2.fastNlMeansDenoising(img)
                if equalize: clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img  = clahe.apply(img)
                #img = cv2.equalizeHist(img)
            elif depth==3:
                if denoise: img = cv2.fastNlMeansDenoisingColored(img)
                if equalize: 
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

    @staticmethod
    def transform(proto, intensity=1.0):
        height, width, *rest = proto.shape
        
        theta = 15. * intensity # Default rotation +/- 15 degrees as described in paper by Yann LeCun
        M = cv2.getRotationMatrix2D((width/2, height/2),
            random.uniform(-theta, theta), 1)

        # Rotate
        img = cv2.warpAffine(proto, M, (width, height), borderMode=cv2.BORDER_REPLICATE)

        # Perpective transformation
        d = width * 0.2 * intensity
        rect = np.array([
            [random.uniform(-d, d), random.uniform(-d, d)],
            [width + random.uniform(-d, d), random.uniform(-d, d)],
            [width + random.uniform(-d, d), height + random.uniform(-d, d)],
            [random.uniform(-d, d), height + random.uniform(-d, d)]], dtype = "float32")
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        img = cv2.warpPerspective(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)

        return img

    @staticmethod
    def transform_crop(rect, image, r_intensity=1.0, p_intensity=1.0, g_intensity=0.):
        height, width, *rest = image.shape
        (x, y, w, h) = rect
        
        theta = 15. * r_intensity # Default rotation +/- 15 degrees as described in paper by Yann LeCun
        M = cv2.getRotationMatrix2D((x+w/2, y+h/2),
            random.triangular(-theta, theta), 1)

        if g_intensity>0.:
            # Weighted Gaussian noise for randomized lighting condition
            gaussian = np.random.random((height, width, 1)).astype(np.float)
            gaussian = np.repeat(gaussian[:, :], 3, axis=2)
            g_weight = 0.25 * g_intensity
            image = (cv2.addWeighted(image.astype(dtype=np.float), (1. - g_weight), g_weight * gaussian, g_weight, 0)).astype(np.uint8)

        # Rotate
        transformed = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REPLICATE)

        # Perpective transformation
        d = w * 0.2 * p_intensity
        rect = np.array([
            [x + random.triangular(-d, d), y + random.triangular(-d, d)],
            [x + w + random.triangular(-d, d), y + random.triangular(-d, d)],
            [x + w + random.triangular(-d, d), y + h + random.triangular(-d, d)],
            [x + random.triangular(-d, d), y + h + random.triangular(-d, d)]], dtype = "float32")
        dst = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        if p_intensity==0:
            transformed = cv2.warpPerspective(transformed, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        else:
            transformed = cv2.warpPerspective(transformed, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        return transformed

    @staticmethod
    def hash(img):
        h = hashlib.new('ripemd160')
        h.update(img.tobytes())
        return h.hexdigest()

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
        block_size = np.copy(shape)
        if len(shape) < 3: shape = (shape,)+(3,)
        #print('ViewportManager::open()', shape)
        shape[0] *= blocks[0]
        shape[1] *= blocks[1]
        if wname not in self.viewports:
            cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
        self.viewports[wname] = {
            'canvas': np.zeros(shape, dtype=np.uint8),
            'block_size': block_size
        }
        cv2.resizeWindow(wname, shape[1], shape[0])
        return self.viewports[wname]['canvas']

    def put(self, wname, img, block):
        canvas = self.viewports[wname]['canvas']
        block_size = self.viewports[wname]['block_size']
        height, width, *rest = img.shape
        #print('put', block_size, img.shape)
        if len(rest):
            canvas[block[0]*block_size[0]:block[0]*block_size[0]+height, block[1]*block_size[1]:block[1]*block_size[1]+width, :] = img[:,:,:]
        else:
            canvas[block[0]*block_size[0]:block[0]*block_size[0]+height, block[1]*block_size[1]:block[1]*block_size[1]+width, :] = img.reshape((height, width, 1))[:,:,:]

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