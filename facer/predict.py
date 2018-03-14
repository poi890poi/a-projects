from shared.utilities import *
from shared.models import *
from shared.dataset import *
from face.dlibdetect import FaceDetector

import os.path
import pprint
import time

import threading
import base64

import cv2
import numpy as np
import tensorflow as tf

shape_raw = (48, 48, 3)
shape_flat = (np.prod(shape_raw),)
shape_image = (12, 12, 3)
n_class = 2
data_size = 2000

def get_data():
    # Hyperparameters
    prep_denoise = False
    prep_equalize = False

    train_count = 0
    val_count = data_size
    total_count = train_count + val_count

    train_positive_count = train_count//2
    train_negative_count = train_count - train_positive_count
    val_positive_count = val_count//2
    val_negative_count = val_count - val_positive_count
    positive_count = train_positive_count + val_positive_count
    negative_count = train_negative_count + val_negative_count

    a = np.zeros(shape=(total_count,)+shape_raw, dtype=np.float32) # Data
    b = np.full(shape=[total_count, n_class], fill_value=-1, dtype=np.int) # Labels
    c = np.zeros(shape=[total_count, n_class], dtype=np.int) # Label assignment
    c[:] = [1, 0]
    c[0:train_positive_count] = [0, 1]
    c[train_count:train_count+val_positive_count] = [0, 1]

    train_data = a[0:train_count, :, :, :]
    train_labels = b[0:train_count, :]
    val_data = a[train_count:total_count, :, :, :]
    val_labels = b[train_count:total_count, :]

    # Load positive data from dataset
    count = positive_count
    face_index = 0
    while count:
        f = DirectoryWalker().get_a_file(directory='../data/face/val/positive', filters=['.jpg'])
        img = cv2.imread(f.path, 1)
        if img is None:
            continue
        src_shape = img.shape
        height, width, *rest = img.shape
        (x, y, w, h) = ImageUtilities.rect_fit_ar([0, 0, width, height], [0, 0, width, height], 1, mrate=1., crop=True)
        if w>0 and h>0:
            pass
        else:
            continue

        #face = img[y:y+h, x:x+w, :]
        face = ImageUtilities.transform_crop((x, y, w, h), img, r_intensity=0., p_intensity=0.)
        face = imresize(face, shape_raw[0:2])
        face = ImageUtilities.preprocess(face, convert_gray=None, equalize=prep_equalize, denoise=prep_denoise)
        #face = tf.image.per_image_standardization(face)
        face = np.array(face, dtype=np.float32)/255

        for i in range(len(b)):
            if np.array_equal(c[i], [0, 1]) and not np.array_equal(b[i], c[i]):
                break
        #print(i, 'positive', imgpath)
        a[i, :, :, :] = face
        b[i] = c[i]

        count -= 1

    # Load negative data
    count = negative_count
    while count:
        f = DirectoryWalker().get_a_file(directory='../data/face/val/negative', filters=['.jpg'])
        img = cv2.imread(f.path, 1)
        height, width, *rest = img.shape
        crop = (height - width)//2
        img = img[crop:crop+width, :, :]
        img = imresize(img, shape_raw[0:2])
        img = ImageUtilities.preprocess(img, convert_gray=None, equalize=prep_equalize, denoise=prep_denoise)
        #img = tf.image.per_image_standardization(img)
        img = np.array(img, dtype=np.float32)/255

        for i in range(len(b)):
            if np.array_equal(c[i], [1, 0]) and not np.array_equal(b[i], c[i]):
                break
        #print(i, 'negative', f.path)
        a[i, :, :, :] = img
        b[i] = c[i]

        count -= 1

    #print(train_data.shape)
    #print(train_labels.shape)
    #print(val_data.shape)
    #print(val_labels.shape)
    return (train_data, train_labels, val_data, val_labels)

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class FaceClassifier(metaclass=Singleton):
    def __init__(self):
        self.model = None

    def init(self, model_dir):
        if self.model is None:
            self.model = FaceCascade({
                'mode': 'INFERENCE',
                'model_dir': '../models/cascade',
                'ckpt_prefix': './server/models/12-net/model.ckpt'
            })

    def detect(self, media):
        timing = dict()
        #print(media)
        img = None
        if 'content' in media:
            bindata = base64.b64decode(media['content'].encode())
            img = cv2.imdecode(np.frombuffer(bindata, np.uint8), 1)

        if img is not None:
            #print(img.shape)
            src_shape = img.shape
            
            time_start = time.time()
            gray = ImageUtilities.preprocess(img, convert_gray=cv2.COLOR_RGB2YCrCb, equalize=False, denoise=False, maxsize=384)
            time_diff = time.time() - time_start
            timing['preprocess'] = time_diff*1000
            #print('preprocess', time_diff)

            processed_shape = gray.shape
            mrate = [processed_shape[0]/src_shape[0], processed_shape[1]/src_shape[1]]

            time_start = time.time()
            rects, landmarks = FaceDetector().detect(gray)
            time_diff = time.time() - time_start
            timing['detect'] = time_diff*1000
            #print('hog+svm detect', time_diff)

            time_start = time.time()
            facelist = list()
            rects_ = list()
            for rect in rects:
                face = None
                (x, y, w, h) = ImageUtilities.rect_to_bb(rect, mrate=mrate)
                height, width, *rest = img.shape
                (x, y, w, h) = ImageUtilities.rect_fit_ar([x, y, w, h], [0, 0, width, height], 1., mrate=1.)
                if w>0 and h>0:
                    face = ImageUtilities.transform_crop((x, y, w, h), img, r_intensity=0., p_intensity=0.)
                    face = imresize(face, shape_raw[0:2])
                    #face = ImageUtilities.preprocess(face, convert_gray=None)
                if face is not None:
                    facelist.append(face)
                    rects_.append([x, y, w, h])
            val_data = np.array(facelist, dtype=np.float32)/255
            reshaped = val_data.reshape((-1,)+shape_flat)
            time_diff = time.time() - time_start
            timing['crop'] = time_diff*1000
            #print('prepare data for cnn', time_diff)

            time_start = time.time()
            feed_dict = {self.model.x: val_data.reshape((-1,)+shape_flat)}
            predictions = self.model.sess.run(self.model.y, feed_dict)
            time_diff = time.time() - time_start
            timing['cnn'] = time_diff*1000
            #print('cnn classify', time_diff, len(facelist))
            #print('predictions', predictions)

            predictions_ = list()
            for p in predictions:
                predictions_.append(p.tolist())

            return (rects_, predictions_, timing)
            