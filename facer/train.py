from shared.utilities import *
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
    train_count = 96
    val_count = 32
    sample_count = train_count+val_count

    model = TensorflowModel('afanet')
    classifier = model.get_estimator()

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
        for i in range(train_count):
            f = DirectoryWalker().get_a_file(directory='../data/face/processed/positive', filters=['.jpg'])
            image = cv2.imread(f.path, 1)
            image = ImageUtilities.preprocess(image)
            image = np.array(image, dtype=np.float32)/255
            a[randomize[rindex], :, :, :] = image
            b[randomize[rindex], :] = 1
            rindex += 1
        for i in range(val_count):
            f = DirectoryWalker().get_a_file(directory='../data/face/processed/negative', filters=['.jpg'])
            image = cv2.imread(f.path, 1)
            image = ImageUtilities.preprocess(image)
            image = np.array(image, dtype=np.float32)/255
            a[randomize[rindex], :, :, :] = image
            rindex += 1

        print('train_data', train_data.shape)
        print('train_labels', train_labels.shape)
        print('val_data', val_data.shape)
        print('val_labels', val_labels.shape)

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': train_data},
            y=train_labels,
            batch_size=8,
            num_epochs=None,
            shuffle=True)
        classifier.train(
            input_fn=train_input_fn,
            steps=256,
            hooks=[logging_hook])

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': val_data},
            y=val_labels,
            num_epochs=1,
            shuffle=False)
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(layers)