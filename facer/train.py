from shared.utilities import *
from shared.models import *
from shared.dataset import *
from face.dlibdetect import FaceDetector

import os.path
import pprint
import time

import cv2
import numpy as np
import tensorflow as tf

shape_raw = (48, 48, 3)
shape_flat = (np.prod(shape_raw),)
shape_image = (12, 12, 3)
n_class = 2
batch_size = 80
epochs = 100
steps = 200
learn_rate = 0.0005

def get_data():
    # Hyperparameters
    prep_denoise = False
    prep_equalize = False

    train_count = batch_size
    val_count = batch_size
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
    data_dir = '../data/face/sof/images'
    dset = SoF()
    dset.load_annotations('../data/face/sof/images/metadata.mat', 'sof')
    count = positive_count
    face_index = 0
    while count:
        anno = dset.get_face(face_index)
        face_index += 1
        if anno is None:
            face_index = 0
            continue
        #print(anno)
        filename = '_'.join((anno['file_prefix'], 'e0_nl_o'))
        filename = '.'.join((filename, 'jpg'))
        imgpath = os.path.normpath(os.path.join(data_dir, filename))

        img = cv2.imread(imgpath, 1)
        if img is None:
            continue
        src_shape = img.shape
        height, width, *rest = img.shape
        (x, y, w, h) = ImageUtilities.rect_fit_ar(anno['rect'].astype(dtype=np.int), [0, 0, width, height], 1, mrate=1.)
        if w>0 and h>0:
            pass
        else:
            continue

        #face = img[y:y+h, x:x+w, :]
        face = ImageUtilities.transform_crop((x, y, w, h), img, r_intensity=1.0, p_intensity=0.)
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
        f = DirectoryWalker().get_a_file(directory='../data/face/processed/negative', filters=['.jpg'])
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

def train(args):
    log_dir = '../models/cascade'
    """if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)"""

    sess = tf.InteractiveSession()
 
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, np.prod(shape_raw)], name='train_data')
        y_ = tf.placeholder(tf.float32, [None, n_class], name='labels')
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, (-1,)+shape_raw)
        tf.summary.image('input', image_shaped_input, batch_size)
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

    # 12-net
    input_12 = tf.image.resize_bilinear(image_shaped_input, (12, 12))
    # Convolutions
    with tf.variable_scope('n12.conv1'):
        conv12_1 = conv_relu(input_12, kernel_size=3, depth=16)
        pool12_1 = pool(conv12_1, size=3, stride=2)
    shape = pool12_1.get_shape().as_list()
    flatten12 = tf.reshape(pool12_1, [-1, shape[1] * shape[2] * shape[3]])
    with tf.variable_scope('n12.fc1'):
        fc12_1 = fully_connected_relu(flatten12, size=16)
        print('net-12 fc', fc12_1)
    """with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(fc1, keep_prob)
    with tf.variable_scope('out'):
        y = fully_connected(dropped, size=n_class)"""

    # 24-net
    input_24 = tf.image.resize_bilinear(image_shaped_input, (24, 24))
    # Convolutions
    with tf.variable_scope('n24.conv1'):
        conv24_1 = conv_relu(input_24, kernel_size=5, depth=64)
        pool24_1 = pool(conv24_1, size=3, stride=2)
    shape = pool24_1.get_shape().as_list()
    flatten24 = tf.reshape(pool24_1, [-1, shape[1] * shape[2] * shape[3]])
    print('net-24 flatten', flatten24)
    with tf.variable_scope('n24.fc1'):
        fc24_1 = fully_connected_relu(flatten24, size=128)
    
    fc24_concat = tf.concat([fc24_1, fc12_1], 1)
    print('net-24 fc', fc24_1)
    print('net-24 concat', fc24_concat)

    """with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(fc24_concat, keep_prob)
    with tf.variable_scope('out'):
        y = fully_connected(dropped, size=n_class)"""

    # 48-net
    # Convolutions
    with tf.variable_scope('n48.conv1'):
        conv48_1 = conv_relu(image_shaped_input, kernel_size=5, depth=64)
        pool48_1 = pool(conv48_1, size=3, stride=2)
        # Normalize, region=9
    with tf.variable_scope('n48.conv2'):
        conv48_2 = conv_relu(pool48_1, kernel_size=5, depth=64)
        # Normalize, region=9
        pool48_2 = pool(conv48_2, size=3, stride=2)
    shape = pool48_2.get_shape().as_list()
    flatten48 = tf.reshape(pool48_2, [-1, shape[1] * shape[2] * shape[3]])
    with tf.variable_scope('n48.fc1'):
        fc48_1 = fully_connected_relu(flatten48, size=256)
    
    fc48_concat = tf.concat([fc48_1, fc24_concat], 1)
    print('net-48 concat', fc48_concat)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(fc48_concat, keep_prob)
    with tf.variable_scope('out'):
        y = fully_connected(dropped, size=n_class)

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

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learn_rate).minimize(
            cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir+'/test')
    tf.global_variables_initializer().run()

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries
    train_data, train_labels, val_data, val_labels = get_data()

    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            xs = train_data.reshape((-1,)+shape_flat)
            ys = train_labels
            k = 0.5
        else:
            xs = val_data.reshape((-1,)+shape_flat)
            ys = val_labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    forward_time = 0
    forward_count = 0
    for i in range(epochs*steps):
        if i % 10 == 0:  # Record summaries and test-set accuracy
            time_start = time.time()
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            time_diff = time.time() - time_start
            forward_time += time_diff
            forward_count += len(val_data)
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s, time/sample: %s' % (i, acc, forward_time/forward_count))
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                    feed_dict=feed_dict(True),
                                    options=run_options,
                                    run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
            if i % steps == steps-1:
                print('Get new data')
                print()
                train_data, train_labels, val_data, val_labels = get_data()
    train_writer.close()
    test_writer.close()

def run(args):
    print('call train instead')
