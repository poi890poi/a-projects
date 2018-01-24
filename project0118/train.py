import argparse
import os.path
import os
from pathlib import Path
import random

import cv2
import numpy as np

from datagen import get_mutations, create_empty_directory
from uuid import uuid4

from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, Embedding
import keras.utils

print(keras.__version__)
print('initialized')

ARGS = None

def preview_input(class_ids):
    for classid in list(class_ids.keys()):
        outdir = os.path.normpath(os.path.join(ARGS.dest, str(classid)))
        create_empty_directory(outdir)
        sample_count = 10    
        srcdir = os.path.normpath(os.path.join(ARGS.src, str(classid)))
        pathlist = Path(srcdir).glob('**/*')
        for path in pathlist:
            path_in_str = str(path)
            mutations = get_mutations(path_in_str, 1)
            for mutation in mutations:
                outpath = os.path.normpath(os.path.join(outdir, str(uuid4()) + '.jpg'))
                cv2.imwrite(outpath, mutation)
                mutation = cv2.normalize(mutation.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            sample_count -= 1
            if sample_count==0: break

def get_model(input_shape, num_classes, model_dir, reset):
    model_path = os.path.join(model_dir, 'lenet5.json')
    model = None
    if os.path.exists(model_path) and not reset:
        # Load model from file
        with open(model_path, 'r') as model_file:
            model_json = model_file.read()
            model = model_from_json(model_json)
    else:
        # Define LeNet-5 model
        model = Sequential()
        model.add(Conv2D(108, (5, 5), activation = 'tanh', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(200, (5, 5), activation = 'tanh', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(100, activation = 'tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation = 'tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation = 'softmax'))

        model_json = model.to_json()
        with open(model_path, 'w') as model_file:
            model_file.write(model_json)

    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

def train_or_load(model, input_shape, class_ids, model_dir, args):
    num_classes = len(class_ids)

    weights_path = os.path.join(model_dir, 'lenet5.weights')
    if os.path.exists(weights_path) and not args.reset:
        # Load weights from file
        model.load_weights(weights_path)
    else:
        for i in range(args.iterations):
            # Prepare training data
            if args.dummy:
                # Generate random dummy data for verification of model definition
                data = np.random.random((1000,)+input_shape)
                labels = np.random.randint(num_classes, size=(1000, 1))
            else:
                # Load trainging data from filesystem
                print(num_classes, args.samples_per_class)
                num_samples = num_classes * args.samples_per_class
                data = np.zeros((num_samples,)+input_shape)
                labels = np.zeros((num_samples, 1))
                data_index = 0
                print('data shape', data.shape)
                for class_id in class_ids:
                    srcdir = os.path.normpath(os.path.join(os.path.normpath(args.src), str(class_id)))
                    pathlist = os.listdir(srcdir)
                    random.shuffle(pathlist)
                    for filename in pathlist[:args.samples_per_class]:
                        sample_path = os.path.normpath(os.path.join(srcdir, filename))
                        sample = get_mutations(sample_path, 1)[0]
                        sample = np.array(cv2.normalize(sample.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)).reshape(input_shape)
                        data[data_index] = sample
                        labels[data_index] = class_id
                        data_index += 1
                        if data_index%500:
                            print(data_index, '/', num_samples)
            one_hot_labels = keras.utils.to_categorical(labels, num_classes=num_classes) # Convert labels to categorical one-hot encoding

            # Train the model
            model.fit(data, one_hot_labels, batch_size=32, epochs=args.epoch, verbose=1)

        model.save_weights(weights_path)
    
def main():
    print('run')
    class_ids = list()
    srcdir = os.path.normpath(ARGS.src)
    pathlist = os.listdir(srcdir)
    for path in pathlist:
        class_ids.append(int(path))

    # Testing getting image mutations
    if ARGS.preview:
        preview_input(class_ids)
        return

    num_classes = len(class_ids)
    print('list', class_ids)

    input_shape = (ARGS.dimension, ARGS.dimension, 1)

    model_dir = os.path.normpath(ARGS.model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, 'lenet5.json')

    model = get_model(input_shape, num_classes, model_dir, ARGS.reset)

    train_or_load(model, input_shape, class_ids, model_dir, ARGS)

    # Load testing data
    batch_size = 500
    data = np.zeros((batch_size,)+input_shape)
    data_index = 0
    total_count = 0
    pathlist = Path(os.path.normpath(ARGS.test_dir)).glob('**/*.ppm')
    file_list = list()
    class_list = list()
    errors = 0
    for path in pathlist:
        imgpath = str(path)
        head, tail = os.path.split(imgpath)
        head, tail = os.path.split(head)
        class_id = int(tail)
        file_list.append(imgpath)
        class_list.append(class_id)
        sample = cv2.imread(imgpath, 0)
        sample = np.array(cv2.normalize(sample.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)).reshape(input_shape)
        data[data_index] = sample
        data_index += 1
        total_count += 1
        if data_index==500:
            predictions = model.predict_classes(data)
            predictions = np.column_stack((np.array(file_list), np.array(predictions), np.array(class_list)))
            #print(predictions)
            for file_path, prediction, class_id in predictions:
                if prediction!=class_id:
                    errors += 1
                    print(file_path, prediction, class_id)
                    print(errors/total_count*100, '%')
            file_list = list()
            class_list = list()
            data_index = 0

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Clean up images and transform to generate more samples""")
    parser.add_argument(
        '--src',
        type=str,
        default='../../data/GTSRB/processed/train',
        help='Path to source directory of training images.'
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        default='../../data/GTSRB/processed/test',
        help='Path to directory of test images.'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='../../models',
        help='Path to directory of models and weights.'
    )
    parser.add_argument(
        '--dimension',
        type=int,
        default=32,
        help='Target dimension of prepared samples.'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=20,
        help='Number of iterations to run keras fit().'
    )
    parser.add_argument(
        '--samples_per_class',
        type=int,
        default=100,
        help='Target dimension of prepared samples.'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=100,
        help='Target dimension of prepared samples.'
    )
    parser.add_argument(
        '--mintensity',
        type=float,
        default=1,
        help='The intensity of mutation.'
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Get random samples for preview.'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Ignore model and weights files and start over.'
    )
    parser.add_argument(
        '--dummy',
        action='store_true',
        help='Use random data to verify that model definition has no error.'
    )
    ARGS, unknown = parser.parse_known_args()
    if unknown:
        print(unknown)
        raise Exception('Unknown argument')
    main()