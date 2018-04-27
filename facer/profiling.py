from mtcnn import detect_face as FaceDetector
from facer.emotion import EmotionClassifier
from shared.utilities import ImageUtilities as imutil
from shared.utilities import DirectoryWalker as dirwalker
from shared.utilities import NumpyEncoder
from shared.alogger import *

from facenet import facenet
import sklearn.metrics, sklearn.preprocessing
import scipy.spatial, scipy.cluster
from sklearn.utils import shuffle

import sys
import time
import numpy as np
import cv2
import tensorflow as tf
import os.path
import pathlib
import math
import uuid
import json
import random
import copy
import base64

from queue import Queue, Empty
import threading
import traceback

#@profile
def profiling(args):
    mtcnn = {}
    mtcnn['session'] = tf.Session()
    #self.__pnet, self.__rnet, self.__onet = FaceDetector.create_mtcnn(self.__mtcnn, None)
    mtcnn['pnet'], mtcnn['rnet'], mtcnn['onet'] = FaceDetector.create_mtcnn(mtcnn['session'], None)

    emoc = EmotionClassifier()
    emoc.build_network(None)

    #facenet_res = facenet.load_model('../models/facenet/20170512-110547.pb') # InceptionResnet V1
    facenet_sqz = facenet.load_model('../models/facenet/20180204-160909') # squeezenet
    