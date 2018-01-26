import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot

import argparse
import os.path
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np
import scipy.signal
import pydot_ng as pydot

from datagen import get_mutations, create_empty_directory
from uuid import uuid4

from keras.models import Model, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, Input, Embedding, concatenate
import keras.callbacks
import keras.utils

print(keras.__version__)
print('initialized')

ARGS = None

def get_model(model_dir, args):
    mptype = args.model_prototype

    model_path = os.path.join(model_dir, mptype + '.json')
    model = None
    if os.path.exists(model_path):
        # Load model from file
        with open(model_path, 'r') as model_file:
            model_json = model_file.read()
            model = model_from_json(model_json)
        print()
        print('Model loaded from', model_path)
    else:
        raise(Exception, 'Model not found')

    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

def load_weights(model, model_dir, args):
    mptype = args.model_prototype

    weights_path = os.path.join(model_dir, mptype + '.weights')
    if os.path.exists(weights_path):
        # Load weights from file
        model.load_weights(weights_path)
        print()
        print('Weights loaded from', weights_path)
    else:
        raise(Exception, 'Weights not found')

def predict(model, data, file_list, class_list):
    predictions = model.predict(data)
    predictions = np.argmax(predictions, axis=1)
    predictions = np.column_stack((np.array(file_list), np.array(predictions), np.array(class_list)))
    return predictions
    
def main():
    class_ids = list()
    srcdir = os.path.normpath(ARGS.test_dir)
    pathlist = os.listdir(srcdir)
    for path in pathlist:
        class_ids.append(int(path))

    num_classes = len(class_ids)

    input_shape = (ARGS.dimension, ARGS.dimension, 1)

    model_dir = os.path.normpath(os.path.join(ARGS.model_dir, ARGS.model_prototype))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model = get_model(model_dir, ARGS)

    load_weights(model, model_dir, ARGS)

    # Load testing data
    print()
    print('Loading test data...')
    batch_size = 512
    data = np.zeros((batch_size,)+input_shape)
    file_list = list()
    class_list = list()
    data_index = 0
    total_count = 0
    pathlist = Path(os.path.normpath(ARGS.test_dir)).glob('**/*.ppm')
    errors = 0
    for path in pathlist:
        imgpath = str(path)
        head, tail = os.path.split(imgpath)
        head, tail = os.path.split(head)
        class_id = int(tail)
        sample = cv2.imread(imgpath, 0)
        sample = np.array(cv2.normalize(sample.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)).reshape(input_shape)
        data[data_index] = sample
        file_list.append(imgpath)
        class_list.append(class_id)
        data_index += 1
        total_count += 1
        if data_index==batch_size:
            predictions = predict(model, data, file_list, class_list)
            for file_path, prediction, class_id in predictions:
                if prediction!=class_id:
                    errors += 1
                    # To-do: Save incorrectly classfied sample...
            file_list = list()
            class_list = list()
            data_index = 0
            print('Error rate:', errors/total_count)

    if data_index:
        predictions = predict(model, data[:data_index], file_list[:data_index], class_list[:data_index])
        for file_path, prediction, class_id in predictions:
            if prediction!=class_id:
                errors += 1
                # To-do: Save incorrectly classfied sample...

    print('Error rate:', errors/total_count)

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Predict""")
    parser.add_argument(
        '--test_dir',
        type=str,
        default='../../data/GTSRB/processed/test',
        help='Path to directory of test images.'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='../../models/lenet',
        help='Path to directory of models and weights.'
    )
    parser.add_argument(
        '--model_prototype',
        type=str,
        default='bn',
        help='The name of model prototype to use with pre-defined hyperparameters.'
    )
    parser.add_argument(
        '--dimension',
        type=int,
        default=32,
        help='Target dimension of prepared samples.'
    )
    ARGS, unknown = parser.parse_known_args()
    if unknown:
        print(unknown)
        raise Exception('Unknown argument')
    main()