from shared.utilities import *
from shared.models import *
from shared.dataset import *
from face.dlibdetect import FaceDetector

import os.path
import pprint
import time
from random import shuffle

import cv2
import numpy as np
import tensorflow as tf

shape_raw = (48, 48, 3)
shape_flat = (np.prod(shape_raw),)
shape_image = (12, 12, 3)
n_class = 2
batch_size = 120
epochs = 20
steps = 2000
learn_rate = 0.0005

filelist = {
    'init': True,
    'positive': list(),
    'negative': list(),
    'pointer': {'positive': 0, 'negative': 0},
}

def prepare_data():
    if filelist['init']:
        while True:
            f = DirectoryWalker().get_a_file(directory='../data/face/train/positive', filters=['.jpg'])
            if f and f.path:
                filelist['positive'].append(f.path)
            else:
                print('positive samples listed', len(filelist['positive']))
                break
        shuffle(filelist['positive'])
        while True:
            f = DirectoryWalker().get_a_file(directory='../data/face/train/negative', filters=['.jpg'])
            if f and f.path:
                filelist['negative'].append(f.path)
            else:
                print('negative samples listed', len(filelist['negative']))
                break
        shuffle(filelist['negative'])
        filelist['init'] = False

    def get_a_file(set):
        if filelist['pointer'][set] >= len(filelist[set]):
            filelist['pointer'][set] = 0
            shuffle(filelist[set])
        index = filelist['pointer'][set]
        filelist['pointer'][set] += 1
        return filelist[set][index]

    # Hyperparameters
    prep_denoise = False
    prep_equalize = False

    train_count = batch_size
    val_count = batch_size//4
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
    print('loading positive samples', count)
    while count:
        imgpath = get_a_file('positive')
        img = cv2.imread(imgpath, 1)
        if img is None:
            continue
        height, width, *rest = img.shape
        (x, y, w, h) = ImageUtilities.rect_fit_ar([0, 0, width, height], [0, 0, width, height], 1, mrate=1., crop=True)
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
    print('loading negative samples', count)
    while count:
        imgpath = get_a_file('negative')
        img = cv2.imread(imgpath, 1)
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
    model = FaceCascade({
        'mode': 'TRAIN',
        'model_dir': '../models/cascade',
        'ckpt_prefix': './server/models/12-net/model.ckpt',
        'batch_size': 120,
        'epochs': 20,
        'steps': 2000,
        'learn_rate': 0.0005,
    })

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries
    train_data, train_labels, val_data, val_labels = prepare_data()
    dataset_train = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    dataset_test = tf.data.Dataset.from_tensor_slices((val_data, val_labels))

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
        return {model.x: xs, model.y_: ys, model.keep_prob: k}

    forward_time = 0
    forward_count = 0
    acc_best = -1
    for i in range(epochs*steps):
        if i % 10 == 0:  # Record summaries and test-set accuracy
            time_start = time.time()
            summary, acc = model.sess.run([model.merged, model.accuracy], feed_dict=feed_dict(False))
            time_diff = time.time() - time_start
            forward_time += time_diff
            forward_count += len(val_data)
            model.test_writer.add_summary(summary, i)

            try:
                if acc >= acc_best:
                    save_path = saver.save(model.sess, model.params['ckpt_prefix'])
                    print('Model saved in path: %s' % save_path)
                    acc_best = acc
            except:
                pass

            print('Accuracy at step %s: %s, time/sample: %s' % (i, acc, forward_time/forward_count))
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = model.sess.run([model.merged, model.train_step],
                                    feed_dict=feed_dict(True),
                                    options=run_options,
                                    run_metadata=run_metadata)
                model.train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                model.train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = model.sess.run([model.merged, model.train_step], feed_dict=feed_dict(True))
                model.train_writer.add_summary(summary, i)
            if i % steps == steps-1:
                print('Get new data')
                print()
                train_data, train_labels, val_data, val_labels = prepare_data()
    model.train_writer.close()
    model.test_writer.close()

def run(args):
    print('call train instead')
