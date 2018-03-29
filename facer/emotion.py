from shared.utilities import *
from shared.models import *
from shared.dataset import *
from face.dlibdetect import FaceDetector
from mtcnn import detect_face as mtcnn_detect
from emotion.emotion_recognition import EmotionRecognition

import os.path
import pprint
import time
import json
import sys

import threading
import base64

import cv2
import numpy as np
import tensorflow as tf

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

class EmotionClassifier:
    def __init__(self):
        self.param = {
            'input_shape': (48, 48, 1),
            'n_class': 7,
        }

    def predict(self, img):
        shape_flat = (np.prod(self.param['input_shape']),)
        val_data = img.reshape((-1,) + shape_flat)
        feed_dict = {self.x: val_data}
        predictions = self.sess.run(self.y, feed_dict)
        return predictions

    def val(self, args):
        input_shape = self.param['input_shape']
        n_class = self.param['n_class']

        count_total = 0
        count_correct = 0
        count_classes = np.zeros((n_class, n_class,), dtype=np.int)

        directory = '../data/face/fer2013/publictest'
        while True:
            f = DirectoryWalker().get_a_file(directory=directory, filters=['.jpg'])
            if f is None: break
            components = os.path.split(f.path)
            label = int(os.path.split(components[0])[1])

            img = cv2.imread(f.path, 0)
            if img is None:break

            path_comp = os.path.normpath(f.path).split('\\')
            label = int(path_comp[-2])

            img = np.array(img, dtype=np.float)/255
            shape_flat = (np.prod(self.param['input_shape']),)
            #print('shape_flat', self.param['input_shape'], shape_flat, img.shape)
            val_data = img.reshape((-1,) + shape_flat)

            feed_dict = {self.x: val_data}
            predictions = self.sess.run(self.y, feed_dict)
            #print('emotion', label, np.argmax(predictions), np.amax(predictions))

            count_total += 1
            if label==np.argmax(predictions):
                count_correct += 1
            count_classes[label][np.argmax(predictions)] += 1

            if count_total%100==99:
                print('precision:', count_correct/count_total)
                row = 0
                col = 0
                str_padding = 10

                sys.stdout.write(''.rjust(str_padding))
                for i, emotion_name in enumerate(EMOTIONS):
                    sys.stdout.write(emotion_name.rjust(str_padding))
                print()

                while row < n_class:
                    sys.stdout.write(EMOTIONS[row].rjust(str_padding))
                    while col < n_class:
                        sys.stdout.write(('%.2f' % (count_classes[row, col]/np.sum(count_classes[row:row+1, :])*100+0.005)).rjust(str_padding))
                        col += 1
                    print()
                    row += 1
                    col = 0
                print()


    def build_network(self, args):
        tf.reset_default_graph() # This line is required when multiple models are used,
                                 # otherwise error 'NotFoundError: Key is_training not found in checkpoint'
                                 # will be encountered when restoring checkpoints
        self.sess = tf.InteractiveSession()

        input_shape = self.param['input_shape']
        n_class = self.param['n_class']

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, np.prod(input_shape)], name='train_data')
            self.y_ = tf.placeholder(tf.float32, [None, n_class], name='labels')
        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(self.x, (-1,)+input_shape)

        def fully_connected(input, size):
            weights = tf.get_variable( 'W', 
                shape = [input.get_shape()[1], size],
                initializer = tf.contrib.layers.xavier_initializer()
            )
            biases = tf.get_variable( 'b',
                shape = [size],
                initializer = tf.constant_initializer(0.0)
            )
            return tf.matmul(input, weights) + biases

        def fully_connected_relu(input, size):
            return tf.nn.relu(fully_connected(input, size))

        def conv_relu(input, kernel_size, depth):
            weights = tf.get_variable( 'W', 
                shape = [kernel_size, kernel_size, input.get_shape()[3], depth],
                initializer = tf.contrib.layers.xavier_initializer()
            )
            biases = tf.get_variable( 'b',
                shape = [depth],
                initializer = tf.constant_initializer(0.0)
            )
            conv = tf.nn.conv2d(input, weights,
                strides = [1, 1, 1, 1], padding = 'SAME')

            return tf.nn.relu(conv + biases)

        def pool(input, size, stride):
            return tf.nn.max_pool(
                input, 
                ksize = [1, size, size, 1], 
                strides = [1, stride, stride, 1], 
                padding = 'SAME'
            )

        # Convolutions
        with tf.variable_scope('Conv2D'):
            input_ = tf.image.resize_bilinear(image_shaped_input, (48, 48))
            _ = conv_relu(input_, kernel_size=5, depth=64)
            _ = pool(_, size=3, stride=2)
        with tf.variable_scope('Conv2D_1'):
            _ = conv_relu(_, kernel_size=5, depth=64)
            _ = pool(_, size=3, stride=2)
        with tf.variable_scope('Conv2D_2'):
            _ = conv_relu(_, kernel_size=4, depth=128)
        with tf.variable_scope('FullyConnected'):
            shape = _.get_shape().as_list()
            _ = tf.reshape(_, [-1, shape[1] * shape[2] * shape[3]])
            _ = fully_connected_relu(_, size=3072)
        with tf.variable_scope('FullyConnected_1'):
            self.y = fully_connected(_, size=n_class)
            self.y = tf.nn.softmax(self.y)

        checkpoint = './emotion/data/emotion_recognition'
        self.saver = tf.train.Saver(filename=checkpoint)
        self.saver.restore(self.sess, checkpoint)
