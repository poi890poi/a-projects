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

def run(args):
    print('do something...')

    # Prepare train data
    train_count = 32
    val_count = 8
    sample_count = train_count + val_count
    positive_count = sample_count//2
    negative_count = sample_count - positive_count

    depth = 5
    out_weights = np.zeros((depth,), dtype=np.float)
    out_weights[0] = 1.

    model = TensorflowModel(args.model)
    classifier = model.get_estimator(out_weights)

    # Set up logging for predictions
    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=8)
    tf.logging.set_verbosity(tf.logging.INFO)

    for epoch in range(256):
        a = np.zeros(shape=[sample_count, 96, 64, 3], dtype=np.float32)
        b = np.zeros(shape=[sample_count, 1], dtype=np.int)
        train_data = a[0:train_count, :, :, :]
        train_labels = b[0:train_count, :]
        val_data = a[train_count:sample_count, :, :, :]
        val_labels = b[train_count:sample_count, :]
        randomize = np.arange(sample_count)
        np.random.shuffle(randomize)
        rindex = 0
        for i in range(positive_count):
            f = DirectoryWalker().get_a_file(directory='../data/face/processed/positive', filters=['.jpg'])
            image = cv2.imread(f.path, 1)
            image = ImageUtilities.preprocess(image)
            image = np.array(image, dtype=np.float32)/255
            a[randomize[rindex], :, :, :] = image
            b[randomize[rindex], :] = 1
            rindex += 1
        for i in range(negative_count):
            f = DirectoryWalker().get_a_file(directory='../data/face/processed/negative', filters=['.jpg'])
            image = cv2.imread(f.path, 1)
            image = ImageUtilities.preprocess(image)
            image = np.array(image, dtype=np.float32)/255
            a[randomize[rindex], :, :, :] = image
            rindex += 1

        print('train_labels', train_labels.shape)
        print('val_labels', val_labels.shape)

        onehot = np.zeros((len(train_labels), depth), dtype=np.float)
        onehot[np.arange(len(train_labels)), train_labels] = 1.
        train_labels = onehot

        onehot = np.zeros((len(val_labels), depth), dtype=np.float)
        onehot[np.arange(len(val_labels)), val_labels] = 1.
        val_labels = onehot

        print('train_data', train_data.shape)
        print('train_labels, one-hot', train_labels.shape)
        print('val_data', val_data.shape)
        print('val_labels, one-hot', val_labels.shape)

        try:
            start_time = time.time()
            # Evaluate the model and print results
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'x': val_data},
                y=val_labels,
                num_epochs=1,
                shuffle=False)
            eval_results = classifier.evaluate(input_fn=eval_input_fn)
            print(eval_results)
            duration = time.time() - start_time
            print('duration', duration, 'per sample', duration/len(val_data))
        except ValueError:
            pass

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': train_data},
            y=train_labels,
            batch_size=32,
            num_epochs=None,
            shuffle=True)
        classifier.train(
            input_fn=train_input_fn,
            steps=4096,
            hooks=[logging_hook])

    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(layers)