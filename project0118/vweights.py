import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot

import argparse
import os.path
import os
from pathlib import Path
import random
import time
import pickle
import sys

import cv2
import numpy as np
import scipy.signal
import skimage.measure
import pydot_ng as pydot

from datagen import get_mutations, create_empty_directory
from uuid import uuid4

from keras.models import Model, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, Input, Embedding, concatenate
import keras.callbacks
import keras.utils
from keras import backend as K

print(keras.__version__)
print('initialized')

ARGS = None

def get_model(model_dir, args):
    mptype = args.model_prototype

    model_path = os.path.join(model_dir, 'model.json')
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

    weights_path = os.path.join(model_dir, 'weights.keras')
    if os.path.exists(weights_path):
        # Load weights from file
        model.load_weights(weights_path)
        print()
        print('Weights loaded from', weights_path)
    else:
        raise(Exception, 'Weights not found')

def predict(model, data, file_list, class_list, output_dir, file_annotations):
    predictions = model.predict(data)
    predictions = np.argmax(predictions, axis=1)
    predictions = np.column_stack((np.array(file_list), np.array(predictions), np.array(class_list)))

    # Save incorrectly classified images
    errors = 0
    for file_path, prediction, class_id in predictions:
        if prediction!=class_id:
            img = cv2.imread(file_path, 1)
            outpath = os.path.join(output_dir, str(uuid4())+'.jpg')
            cv2.imwrite(outpath, img)
            filename = os.path.split(outpath)[1]
            pickle.dump([filename, int(prediction), int(class_id)], file_annotations, protocol=pickle.HIGHEST_PROTOCOL)
            errors += 1

    return errors
    
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

    outdir = os.path.normpath(os.path.join(model_dir, 'errors'))
    create_empty_directory(outdir)
    annotations_path = os.path.join(outdir, 'annotations.pkl')
    print(annotations_path)
    file_annotations = open(annotations_path, 'wb')
    
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

        filters_dir = os.path.normpath(os.path.join(model_dir, 'filters'))
        create_empty_directory(filters_dir)

        sample = np.array(cv2.normalize(sample.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)).reshape(input_shape)

        print(input_shape)
        output_shape = input_shape[:2]
        print(output_shape)

        conv2 = model.get_layer('conv_2')
        print(conv2.get_weights()[0].shape)
        kernel = conv2.get_weights()[0][:, :, :, 0]
        print(kernel.shape)

        conv1 = model.get_layer('conv_1')
        relu1 = model.get_layer('relu_1')

        for i in range(32):
            k1 = conv1.get_weights()[0][:, :, :, i].reshape((5, 5))
            #sample = sample.reshape(output_shape) # Loaded image
            sample = np.random.random(output_shape) - 0.5 # Random pixels

            convolved1 = scipy.signal.convolve2d(sample, k1, mode='same', boundary='wrap')
            activated1 = convolved1 * (convolved1 > 0)
            pooled1 = skimage.measure.block_reduce(activated1, (2,2), np.max)

            img = np.array(cv2.normalize(convolved1.astype('float'), None, 0.0, 255.0, cv2.NORM_MINMAX))
            cv2.imwrite(os.path.join(filters_dir, str(i).zfill(2)+'-1c-00.jpg'), img)

            img = np.array(cv2.normalize(activated1.astype('float'), None, 0.0, 255.0, cv2.NORM_MINMAX))
            cv2.imwrite(os.path.join(filters_dir, str(i).zfill(2)+'-2a-00.jpg'), img)

            img = np.array(cv2.normalize(pooled1.astype('float'), None, 0.0, 255.0, cv2.NORM_MINMAX))
            cv2.imwrite(os.path.join(filters_dir, str(i).zfill(2)+'-3p-00.jpg'), img)

            for j in range(0):
                k2 = conv2.get_weights()[0][:, :, i, j].reshape((5, 5))

                convolved2 = scipy.signal.convolve2d(convolved1, k2, mode='same', boundary='wrap')
                img = np.array(cv2.normalize(convolved2.astype('float'), None, 0.0, 255.0, cv2.NORM_MINMAX))
                cv2.imwrite('./debug/conv-'+str(i).zfill(2)+'-1-'+str(j).zfill(2)+'.jpg', img)

        sys.exit()

        layer_output = conv1.output
        model.predict([sample])

        sys.exit()

        data[data_index] = sample
        file_list.append(imgpath)
        class_list.append(class_id)
        data_index += 1
        total_count += 1
        if data_index==batch_size:
            errors += predict(model, data, file_list, class_list, outdir, file_annotations)
            file_list = list()
            class_list = list()
            data_index = 0
            print('Error rate:', errors/total_count)

    if data_index:
        errors += predict(model, data[:data_index], file_list[:data_index], class_list[:data_index], outdir, file_annotations)

    pickle.dump(['[stats]', errors, total_count], file_annotations, protocol=pickle.HIGHEST_PROTOCOL)
    file_annotations.close()

    print()
    print('Overall accuracy:', 1 - errors/total_count)

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