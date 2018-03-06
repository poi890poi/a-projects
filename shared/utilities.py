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
import os, stat

import random

"""class TensorflowModel():
    def __init__(self, model):
        self.model = model

    def model_fn(self, features, labels, mode, params=None):
        #input_layer = tf.reshape(features['x'], [-1, 64, 96, 3])
        input_layer = features['x']

        self.layers = list()
        if self.model=='darknet19':
            self.darknet19(input_layer, labels, mode)
        elif self.model=='afanet':
            self.afanet(input_layer, labels, mode)
        elif self.model=='afanet7':
            self.afanet7(input_layer, labels, mode)
        else:
            raise ValueError('Unrecognized model name')

        logits = self.layers[-1]

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            'classes': tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            'probabilities': tf.nn.softmax(logits, name='softmax')
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        #loss = tf.reduce_mean(tf.square(expected - net))
        print('out_weights', params['out_weights'])
        print('labels', labels)
        print('logits', logits)
        loss = tf.abs(labels - tf.cast(logits, tf.float64))*params['out_weights']
        print('loss', loss)
        loss = tf.reduce_mean(tf.abs(labels - tf.cast(logits, tf.float64))*params['out_weights'])
        #loss = tf.reduce_mean(tf.abs(labels - tf.cast(logits, tf.float64)))
        #loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            'accuracy': tf.metrics.mean_squared_error(
                labels=labels, predictions=tf.cast(logits, tf.float64))}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def get_estimator(self, out_weights=None):
        # Create the Estimator
        return tf.estimator.Estimator(
            model_fn=self.model_fn, params={'out_weights': out_weights}, model_dir='../models/'+self.model)

    def afanet7(self, input_layer, labels, mode):
        self.layers.append(self.conv2d(input_layer, 32, kernel_size=[5, 5], strides=[2, 2], activation='leaky_relu'))
        self.layers.append(self.pool2d(None))
        self.layers.append(self.conv2d(None, 96, activation='leaky_relu'))
        self.layers.append(self.pool2d(None))

        self.layers.append(self.conv2d(None, 64, kernel_size=[1, 1], activation='leaky_relu'))
        self.layers.append(self.conv2d(None, 128, activation='leaky_relu'))
        self.layers.append(self.pool2d(None))

        self.layers.append(self.conv2d(None, 100, kernel_size=[1, 1], activation='leaky_relu'))
        self.layers.append(tf.reduce_mean(self.layers[-1], [1, 2], name='avg_pool'))
        
        for i, layer in enumerate(self.layers):
            print(i, layer)

    def afanet(self, input_layer, labels, mode):
        self.layers.append(self.conv2d(input_layer, 32, kernel_size=[5, 5], strides=[2, 2], activation='leaky_relu'))
        self.layers.append(self.pool2d(None))
        self.layers.append(self.conv2d(None, 96, activation='leaky_relu'))
        self.layers.append(self.pool2d(None))

        self.layers.append(self.conv2d(None, 32, kernel_size=[1, 1], activation='leaky_relu'))
        self.layers.append(self.conv2d(None, 64, activation='leaky_relu'))
        self.layers.append(self.conv2d(None, 128, activation='leaky_relu'))
        self.layers.append(self.pool2d(None))

        self.layers.append(self.conv2d(None, 64, kernel_size=[1, 1], activation='leaky_relu'))
        self.layers.append(self.conv2d(None, 128, activation='leaky_relu'))

        self.layers.append(self.conv2d(None, 100, kernel_size=[1, 1], activation='leaky_relu'))
        self.layers.append(tf.reduce_mean(self.layers[-1], [1, 2], name='avg_pool'))
        
        #for i, layer in enumerate(self.layers):
        #    print(i, layer)

    def darknet19(self, input_layer, labels, mode):
        '''
         0 conv     64  7 x 7 / 2   224 x 224 x   3   ->   112 x 112 x  64
         1 max          2 x 2 / 2   112 x 112 x  64   ->    56 x  56 x  64
         2 conv    192  3 x 3 / 1    56 x  56 x  64   ->    56 x  56 x 192
         3 max          2 x 2 / 2    56 x  56 x 192   ->    28 x  28 x 192
         4 conv    128  1 x 1 / 1    28 x  28 x 192   ->    28 x  28 x 128
         5 conv    256  3 x 3 / 1    28 x  28 x 128   ->    28 x  28 x 256
         6 conv    256  1 x 1 / 1    28 x  28 x 256   ->    28 x  28 x 256
         7 conv    512  3 x 3 / 1    28 x  28 x 256   ->    28 x  28 x 512
         8 max          2 x 2 / 2    28 x  28 x 512   ->    14 x  14 x 512
         9 conv    256  1 x 1 / 1    14 x  14 x 512   ->    14 x  14 x 256
        10 conv    512  3 x 3 / 1    14 x  14 x 256   ->    14 x  14 x 512
        11 conv    256  1 x 1 / 1    14 x  14 x 512   ->    14 x  14 x 256
        12 conv    512  3 x 3 / 1    14 x  14 x 256   ->    14 x  14 x 512
        13 conv    256  1 x 1 / 1    14 x  14 x 512   ->    14 x  14 x 256
        14 conv    512  3 x 3 / 1    14 x  14 x 256   ->    14 x  14 x 512
        15 conv    256  1 x 1 / 1    14 x  14 x 512   ->    14 x  14 x 256
        16 conv    512  3 x 3 / 1    14 x  14 x 256   ->    14 x  14 x 512
        17 conv    512  1 x 1 / 1    14 x  14 x 512   ->    14 x  14 x 512
        18 conv   1024  3 x 3 / 1    14 x  14 x 512   ->    14 x  14 x1024
        19 max          2 x 2 / 2    14 x  14 x1024   ->     7 x   7 x1024
        20 conv    512  1 x 1 / 1     7 x   7 x1024   ->     7 x   7 x 512
        21 conv   1024  3 x 3 / 1     7 x   7 x 512   ->     7 x   7 x1024
        22 conv    512  1 x 1 / 1     7 x   7 x1024   ->     7 x   7 x 512
        23 conv   1024  3 x 3 / 1     7 x   7 x 512   ->     7 x   7 x1024
        24 conv   1000  1 x 1 / 1     7 x   7 x1024   ->     7 x   7 x1000
        25 avg                        7 x   7 x1000   ->  1000
        26 softmax                                        1000
        27 cost                                           1000
        '''

        self.layers.append(self.conv2d(input_layer, 64, kernel_size=[7, 7], strides=[2, 2], activation='leaky_relu'))
        self.layers.append(self.pool2d(None))
        self.layers.append(self.conv2d(None, 192, activation='leaky_relu'))
        self.layers.append(self.pool2d(None))

        self.layers.append(self.conv2d(None, 128, kernel_size=[1, 1], activation='leaky_relu'))
        self.layers.append(self.conv2d(None, 256, activation='leaky_relu'))
        self.layers.append(self.conv2d(None, 256, kernel_size=[1, 1], activation='leaky_relu'))
        self.layers.append(self.conv2d(None, 512, activation='leaky_relu'))
        self.layers.append(self.pool2d(None))

        for i in range(4):
            self.layers.append(self.conv2d(None, 256, kernel_size=[1, 1], activation='leaky_relu'))
            self.layers.append(self.conv2d(None, 512, activation='leaky_relu'))
        self.layers.append(self.conv2d(None, 512, kernel_size=[1, 1], activation='leaky_relu'))
        self.layers.append(self.conv2d(None, 1024, activation='leaky_relu'))
        self.layers.append(self.pool2d(None))

        for i in range(2):
            self.layers.append(self.conv2d(None, 512, kernel_size=[1, 1], activation='leaky_relu'))
            self.layers.append(self.conv2d(None, 1024, activation='leaky_relu'))

        self.layers.append(self.conv2d(None, 1000, kernel_size=[1, 1], activation='leaky_relu'))
        print(self.layers[-1].shape)
        #self.layers.append(tf.nn.avg_pool(self.layers[-1], [1, 1, 7, 7], [1, 1, 1, 1], padding='SAME', name='avg_pool'))
        self.layers.append(tf.reduce_mean(self.layers[-1], [1, 2], name='avg_pool'))
        
        for i, layer in enumerate(self.layers):
            print(i, layer)

    def get_scope_name(self, index, suffix):
        return str(index).zfill(2)+'.'+suffix
    
    def conv2d(self, input_layer, num_filters, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu', normalize=True, name=''):
        if input_layer==None:
            input_layer = self.layers[-1]
        if name=='':
            name = self.get_scope_name(len(self.layers), 'conv')
        act_func = tf.nn.relu
        if activation=='leaky_relu':
            act_func = tf.nn.leaky_relu
        conv = tf.layers.conv2d(
            inputs=input_layer,
            filters=num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=act_func,
            name=name)
        if normalize:
            conv = tf.layers.batch_normalization(conv, name=name)
        return conv

    def pool2d(self, inputs, size=[2,2], strides=2, name=''):
        if inputs==None:
            inputs = self.layers[-1]
        if name=='':
            name = self.get_scope_name(len(self.layers), 'max')
        return tf.layers.max_pooling2d(inputs=inputs, pool_size=size, strides=strides, name=name)

    def dense(self, inputs, units, activation='linear', dropout=-1, name=''):
        if inputs==None:
            inputs = self.layers[-1]
        if name=='':
            name = self.get_scope_name(len(self.layers), 'dense')
        act_func = None
        print(inputs.get_shape()[-3:], np.prod(inputs.get_shape()[-3:]))
        flatten = tf.reshape(inputs, [-1, np.prod(inputs.get_shape()[-3:])])
        dense = tf.layers.dense(inputs=flatten, units=units, activation=act_func, name=name)
        if dropout>0:
            dense = tf.layers.dropout(
                inputs=dense, rate=dropout, training=mode == tf.estimator.ModeKeys.TRAIN, name=name)
        return dense"""

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
            #img = cv2.equalizeHist(img)
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