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

def predict(args):
    log_dir = '../models/cascade_hx/20180312'
    """if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)"""

    sess = tf.InteractiveSession()
 
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, np.prod(shape_raw)], name='train_data')
        y_ = tf.placeholder(tf.float32, [None, n_class], name='labels')
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, (-1,)+shape_raw)
        tf.summary.image('input', image_shaped_input, data_size)
        print(image_shaped_input)

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def fully_connected(input, size):
        weights = tf.get_variable( 'weights', 
            shape = [input.get_shape()[1], size],
            initializer = tf.contrib.layers.xavier_initializer()
        )
        biases = tf.get_variable( 'biases',
            shape = [size],
            initializer = tf.constant_initializer(0.0)
        )
        variable_summaries(weights)
        variable_summaries(biases)
        return tf.matmul(input, weights) + biases

    def fully_connected_relu(input, size):
        return tf.nn.relu(fully_connected(input, size))

    def conv_relu(input, kernel_size, depth):
        weights = tf.get_variable( 'weights', 
            shape = [kernel_size, kernel_size, input.get_shape()[3], depth],
            initializer = tf.contrib.layers.xavier_initializer()
        )
        biases = tf.get_variable( 'biases',
            shape = [depth],
            initializer = tf.constant_initializer(0.0)
        )
        print(weights.name)
        print(biases.name)
        variable_summaries(weights)
        variable_summaries(biases)
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

    with tf.variable_scope('12-net'):
        input_12 = tf.image.resize_bilinear(image_shaped_input, (12, 12))
        # Convolutions
        with tf.variable_scope('conv1'):
            conv12_1 = conv_relu(input_12, kernel_size=3, depth=16)
            pool12_1 = pool(conv12_1, size=3, stride=2)
            print(conv12_1.name)
            print(pool12_1.name)
        shape = pool12_1.get_shape().as_list()
        flatten12 = tf.reshape(pool12_1, [-1, shape[1] * shape[2] * shape[3]])
        with tf.variable_scope('fc1'):
            fc12_1 = fully_connected_relu(flatten12, size=16)
            fc_final = fc12_1

    """with tf.variable_scope('24-net'):
        input_24 = tf.image.resize_bilinear(image_shaped_input, (24, 24))
        # Convolutions
        with tf.variable_scope('conv1'):
            conv24_1 = conv_relu(input_24, kernel_size=5, depth=64)
            pool24_1 = pool(conv24_1, size=3, stride=2)
        shape = pool24_1.get_shape().as_list()
        flatten24 = tf.reshape(pool24_1, [-1, shape[1] * shape[2] * shape[3]])
        print('net-24 flatten', flatten24)
        with tf.variable_scope('fc1'):
            fc24_1 = fully_connected_relu(flatten24, size=128)
        
        fc24_concat = tf.concat([fc24_1, fc12_1], 1)
        print('net-24 fc', fc24_1)
        print('net-24 concat', fc24_concat)

    with tf.variable_scope('48-net'):
        # Convolutions
        with tf.variable_scope('conv1'):
            conv48_1 = conv_relu(image_shaped_input, kernel_size=5, depth=64)
            pool48_1 = pool(conv48_1, size=3, stride=2)
            # Normalize, region=9
        with tf.variable_scope('conv2'):
            conv48_2 = conv_relu(pool48_1, kernel_size=5, depth=64)
            # Normalize, region=9
            pool48_2 = pool(conv48_2, size=3, stride=2)
        shape = pool48_2.get_shape().as_list()
        flatten48 = tf.reshape(pool48_2, [-1, shape[1] * shape[2] * shape[3]])
        with tf.variable_scope('fc1'):
            fc48_1 = fully_connected_relu(flatten48, size=256)
        
        fc48_concat = tf.concat([fc48_1, fc24_concat], 1)
        print('net-48 concat', fc48_concat)"""

    with tf.variable_scope('out'):
        y = fully_connected(fc_final, size=n_class)

    with tf.name_scope('cross_entropy'):
        # The raw formulation of cross-entropy,
        #
        # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
        #                               reduction_indices=[1]))
        #
        # can be numerically unstable.
        #
        # So here we use tf.nn.softmax_cross_entropy_with_logits on the
        # raw outputs of the nn_layer above, and then average across
        # the batch.
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter(log_dir+'/test')
    tf.global_variables_initializer().run()

    y = tf.nn.softmax(y)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(filename=log_dir+'/model.ckpt')

    # Restore variables from disk.
    time_start = time.time()
    saver.restore(sess, log_dir+'/model.ckpt')
    time_diff = time.time() - time_start
    print('Model restored.', time_diff)

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries
    train_data, train_labels, val_data, val_labels = get_data()
    dataset_train = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    dataset_test = tf.data.Dataset.from_tensor_slices((val_data, val_labels))

    forward_time = 0
    forward_count = 0
    stats_correct = 0
    stats_positive = 0
    stats_positive_true = 0
    stats_total = 0
    call_time = 0
    call_count = 0
    for j in range(100): # 100 iterations for performance benchmarking
        time_start = time.time()
        feed_dict = {x: val_data.reshape((-1,)+shape_flat)}
        predictions = sess.run(y, feed_dict)
        time_diff = time.time() - time_start
        forward_time += time_diff
        forward_count += len(val_data)

        print('val_data', val_data.shape, val_data.dtype)
        reshaped = val_data.reshape((-1,)+shape_flat)
        print('reshaped', reshaped.shape, reshaped.dtype)
        print('preprocess for cnn', time_diff)

        # Predict with only one sample to get call overhead
        time_start = time.time()
        feed_dict_one = {x: val_data[0:1].reshape((-1,)+shape_flat)}
        print('feed_dict', feed_dict_one)
        p_one = sess.run(y, feed_dict_one)
        time_diff_one = time.time() - time_start
        call_time += time_diff - ((time_diff - time_diff_one) + (time_diff-time_diff_one)/(len(val_data)-1))
        call_count += 1

        if j==0:
            i = 0
            for p in predictions:
                val = val_labels[i]

                if val[1] > val[0]:
                    img = (val_data[i]*255).astype(dtype=np.uint8)
                    filename = '../models/cascade/val_result/positive/'+ImageUtilities.hash(img)+'.jpg'
                    cv2.imwrite(filename, img)
                    stats_positive += 1

                if (p[0]>p[1] and val[0]>val[1]) or (p[1]>p[0] and val[1]>val[0]):
                    stats_correct += 1
                    if val[1] > val[0]:
                        img = (val_data[i]*255).astype(dtype=np.uint8)
                        if p[1] > p[0]:
                            stats_positive_true += 1
                            filename = '../models/cascade/val_result/positive_true/'+ImageUtilities.hash(img)+'.jpg'
                            cv2.imwrite(filename, img)
                else:
                    img = (val_data[i]*255).astype(dtype=np.uint8)
                    if p[1] > p[0]:
                        filename = '../models/cascade/val_result/false_positive/'+ImageUtilities.hash(img)+'.jpg'
                        cv2.imwrite(filename, img)
                    else:
                        filename = '../models/cascade/val_result/false_negative/'+ImageUtilities.hash(img)+'.jpg'
                        cv2.imwrite(filename, img)

                stats_total += 1
                i += 1
    
    print('time:', forward_time*100/forward_count, 'ms per 100 samples')
    print('overhead:', call_time*100/call_count, 'ms per 100 call')
    print('precision:', stats_correct/stats_total)
    print('recall:', stats_positive_true/stats_positive)

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class FaceClassifier(metaclass=Singleton):
    def init(self, model_dir):
        print('Restoring TF model...', model_dir)
        try:
            print(self.sess, threading.current_thread().ident)
            return
        except AttributeError:
            pass # Continue intializing...

        self.sess = tf.InteractiveSession()
    
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, np.prod(shape_raw)], name='train_data')
            y_ = tf.placeholder(tf.float32, [None, n_class], name='labels')
        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(self.x, (-1,)+shape_raw)
            tf.summary.image('input', image_shaped_input, data_size)

        def fully_connected(input, size):
            weights = tf.get_variable( 'weights', 
                shape = [input.get_shape()[1], size],
                initializer = tf.contrib.layers.xavier_initializer()
            )
            biases = tf.get_variable( 'biases',
                shape = [size],
                initializer = tf.constant_initializer(0.0)
            )
            return tf.matmul(input, weights) + biases

        def fully_connected_relu(input, size):
            return tf.nn.relu(fully_connected(input, size))

        def conv_relu(input, kernel_size, depth):
            weights = tf.get_variable( 'weights', 
                shape = [kernel_size, kernel_size, input.get_shape()[3], depth],
                initializer = tf.contrib.layers.xavier_initializer()
            )
            biases = tf.get_variable( 'biases',
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

        with tf.variable_scope('12-net'):
            input_12 = tf.image.resize_bilinear(image_shaped_input, (12, 12))
            # Convolutions
            with tf.variable_scope('conv1'):
                conv12_1 = conv_relu(input_12, kernel_size=3, depth=16)
                pool12_1 = pool(conv12_1, size=3, stride=2)
            shape = pool12_1.get_shape().as_list()
            flatten12 = tf.reshape(pool12_1, [-1, shape[1] * shape[2] * shape[3]])
            with tf.variable_scope('fc1'):
                fc12_1 = fully_connected_relu(flatten12, size=16)
                fc_final = fc12_1

        """with tf.variable_scope('24-net'):
            input_24 = tf.image.resize_bilinear(image_shaped_input, (24, 24))
            # Convolutions
            with tf.variable_scope('conv1'):
                conv24_1 = conv_relu(input_24, kernel_size=5, depth=64)
                pool24_1 = pool(conv24_1, size=3, stride=2)
            shape = pool24_1.get_shape().as_list()
            flatten24 = tf.reshape(pool24_1, [-1, shape[1] * shape[2] * shape[3]])
            print('net-24 flatten', flatten24)
            with tf.variable_scope('fc1'):
                fc24_1 = fully_connected_relu(flatten24, size=128)
            
            fc24_concat = tf.concat([fc24_1, fc12_1], 1)
            print('net-24 fc', fc24_1)
            print('net-24 concat', fc24_concat)

        with tf.variable_scope('48-net'):
            # Convolutions
            with tf.variable_scope('conv1'):
                conv48_1 = conv_relu(image_shaped_input, kernel_size=5, depth=64)
                pool48_1 = pool(conv48_1, size=3, stride=2)
                # Normalize, region=9
            with tf.variable_scope('conv2'):
                conv48_2 = conv_relu(pool48_1, kernel_size=5, depth=64)
                # Normalize, region=9
                pool48_2 = pool(conv48_2, size=3, stride=2)
            shape = pool48_2.get_shape().as_list()
            flatten48 = tf.reshape(pool48_2, [-1, shape[1] * shape[2] * shape[3]])
            with tf.variable_scope('fc1'):
                fc48_1 = fully_connected_relu(flatten48, size=256)
            
            fc48_concat = tf.concat([fc48_1, fc24_concat], 1)
            print('net-48 concat', fc48_concat)"""

        with tf.variable_scope('out'):
            self.y = fully_connected(fc_final, size=n_class)

        # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        #tf.global_variables_initializer().run()

        self.y = tf.nn.softmax(self.y)

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver(filename=model_dir+'/model.ckpt')

        # Restore variables from disk.
        time_start = time.time()
        self.saver.restore(self.sess, model_dir+'/model.ckpt')
        time_diff = time.time() - time_start
        print('Model restored.', time_diff)

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
            feed_dict = {self.x: val_data.reshape((-1,)+shape_flat)}
            predictions = self.sess.run(self.y, feed_dict)
            time_diff = time.time() - time_start
            timing['cnn'] = time_diff*1000
            #print('cnn classify', time_diff, len(facelist))
            #print('predictions', predictions)

            predictions_ = list()
            for p in predictions:
                predictions_.append(p.tolist())

            return (rects_, predictions_, timing)
            