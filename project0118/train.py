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

class TrainHistory(keras.callbacks.Callback):
    def __init__(self, args):
        self.sampling_window = 16
        self.sampling_count = self.sampling_window
        self.max_len = 1024
        self.epoch = 1
        self.epochs = []
        self.losses = []
        self.acc = []

        mptype = args.model_prototype
        model_dir = os.path.normpath(args.model_dir)
        self.graph_path = os.path.join(model_dir, mptype + '-history.png')

    def update_graph(self):
        try:
            '''if len(self.epochs) >= self.max_len:
                self.epochs = list(np.array(self.epochs)[::2])[:int(self.max_len/2)]
                self.losses = list(scipy.signal.resample(self.losses, int(self.max_len/2)))
                self.acc = list(scipy.signal.resample(self.acc, int(self.max_len/2)))'''

            epochs = np.array(self.epochs)
            losses = np.array(self.losses)
            acc = np.array(self.acc)

            fig, ax = matplotlib.pyplot.subplots()
            x_axis = range(len(losses))
            ax.plot(x_axis, losses, 'r:', label='Loss', linewidth=1.0)
            ax.plot(x_axis, acc, 'g', label='Accuracy', linewidth=1.0)

            #legend = ax.legend(loc='upper right', shadow=False, fontsize='x-medium')
            #legend.get_frame().set_facecolor('#BBFFCC')
            ax.set(xlabel='epochs', ylabel='',
                title='Training history')
            ax.grid()

            fig.savefig(self.graph_path)
        except Exception:
            pass

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')
        acc = logs.get('acc')
        if self.sampling_count>=self.sampling_window:
            # Expand history list
            self.losses.append(logs.get('loss'))
            self.acc.append(logs.get('acc'))
            self.sampling_count = 1
        else:
            # Valculate averaged data
            pt = len(self.losses) - 1
            self.losses[pt] = (self.losses[pt] * self.sampling_count + loss) / (self.sampling_count+1)
            self.acc[pt] = (self.acc[pt] * self.sampling_count + acc) / (self.sampling_count+1)
            self.sampling_count += 1
            self.update_graph()

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

def get_model(input_shape, num_classes, model_dir, args):
    model_prototypes = {
        'ss' : [32, 64, 2, 4, 400, 400, 0],
        'mm' : [32, 64, 2, 4, 400, 400, 2],
        'bn' : [32, 64, 2, 4, 400, 400, 3],
    }
    FLAG_BATCHNORMALIZATION = 1
    FLAG_MULTISCALE = 2
    FLAG_EARLYSTOP = 4
    mptype = args.model_prototype
    f1 = model_prototypes[mptype][0]
    f2 = model_prototypes[mptype][1]
    p1 = model_prototypes[mptype][2]
    p2 = model_prototypes[mptype][3]
    fc1 = model_prototypes[mptype][4]
    fc2 = model_prototypes[mptype][5]
    flags = model_prototypes[mptype][6]

    model_path = os.path.join(model_dir, mptype + '.json')
    model = None
    if os.path.exists(model_path) and not args.reset:
        # Load model from file
        with open(model_path, 'r') as model_file:
            model_json = model_file.read()
            model = model_from_json(model_json)
        print()
        print('Model loaded from', model_path)
    else:
        # Define LeNet multi-scale model
        in_raw = Input(shape=input_shape) # Raw images as source input
        x = Conv2D(f1, (5, 5), kernel_initializer='glorot_normal', input_shape=input_shape, name='conv_1')(in_raw)
        x = BatchNormalization()(x)
        x = Activation('relu', name='relu_1')(x)
        x = MaxPooling2D(pool_size=(p1, p1), name='maxpool_1')(x)

        # Define output of stage-1
        in_s1 = Flatten()(x)

        # Begin of stage-2
        x = Conv2D(f2, (5, 5), kernel_initializer='glorot_normal', input_shape=input_shape, name='conv_2')(x)
        if (flags&FLAG_BATCHNORMALIZATION): x = BatchNormalization()(x)
        x = Activation('relu', name='relu_2')(x)
        x = MaxPooling2D(pool_size=(p2, p2), name='maxpool_2')(x)
        x = Flatten()(x)

        # Concatenate outputs from stage-1 and stage-2
        if (flags&FLAG_MULTISCALE): x = concatenate([x, in_s1])

        # Use 2 fully-connected layers
        x = Dense(fc1, name='fc_1')(x)
        if (flags&FLAG_BATCHNORMALIZATION): x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(fc2, name='fc_2')(x)
        if (flags&FLAG_BATCHNORMALIZATION): x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes)(x)
        if (flags&FLAG_BATCHNORMALIZATION): x = BatchNormalization()(x)
        predictions = Activation('softmax', name='softmax')(x)

        model = Model(inputs=in_raw, outputs=predictions)

        model_json = model.to_json()
        with open(model_path, 'w') as model_file:
            model_file.write(model_json)

    # Save model plot
    graph_path = os.path.join(model_dir, mptype + '-model.png')
    keras.utils.plot_model(model, to_file=graph_path)

    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


def train_or_load(model, input_shape, class_ids, model_dir, args):
    train_parameters = {
        'ss' : [1024, 64, 8],
        'mm' : [1024, 64, 8],
        'bn' : [1024, 64, 8],
    }
    mptype = args.model_prototype
    iterations = train_parameters[mptype][0]
    batch_size = train_parameters[mptype][1]
    epochs = train_parameters[mptype][2]

    num_classes = len(class_ids)

    weights_path = os.path.join(model_dir, mptype + '.weights')
    if os.path.exists(weights_path) and not args.reset:
        # Load weights from file
        model.load_weights(weights_path)
        print()
        print('Weights loaded from', weights_path)
    else:
        if args.dummy:
            # Generate random dummy data for verification of model definition
            data = np.random.random((1000,)+input_shape)
            labels = np.random.randint(num_classes, size=(1000, 1))
            one_hot_labels = keras.utils.to_categorical(labels, num_classes=num_classes) # Convert labels to categorical one-hot encoding
            model.fit(data, one_hot_labels, batch_size=batch_size, epochs=args.epoch, verbose=1)
        else:
            # Get list of proto-samples and shuffle them
            print()
            print('Getting sample file list...')
            now = int(time.time())
            samples = list()
            srcdir = os.path.normpath(args.src)
            pathlist = sorted(Path(srcdir).glob('**/*.ppm'))
            for path in pathlist:
                # A image folder with .csv file (annotations)
                path_in_str = str(path)
                sample_dir, sample_filename = os.path.split(path_in_str)
                class_id = os.path.split(sample_dir)[1]
                samples.append([sample_filename, class_id, path_in_str])
            random.shuffle(samples)

            validation_data = []
            validation_labels = []
            if args.validation:
                print()
                print('Shuffling validation data...')
                # Sampling random data from directory of test images
                pathlist = Path(os.path.normpath(args.validation)).glob('**/*.ppm')
                entry_list = list()
                for path in pathlist:
                    imgpath = str(path)
                    head, tail = os.path.split(imgpath)
                    head, tail = os.path.split(head)
                    class_id = int(tail)
                    entry_list.append([class_id, imgpath])
                random.shuffle(entry_list)

                # Load validation data
                print('Loading validation data...')
                validation_size = batch_size
                entry_list = entry_list[:validation_size]
                validation_data = np.zeros((validation_size,)+input_shape)
                labels = np.zeros((validation_size, 1))
                data_index = 0
                for data_entry in entry_list:
                    class_id = data_entry[0]
                    imgpath = data_entry[1]
                    sample = cv2.imread(imgpath, 0)
                    sample = np.array(cv2.normalize(sample.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)).reshape(input_shape)
                    validation_data[data_index] = sample
                    labels[data_index] = class_id
                    data_index += 1
                validation_labels = keras.utils.to_categorical(labels, num_classes=num_classes) # Convert labels to categorical one-hot encoding

            # Iterations of training
            sample_offset = 0
            history = TrainHistory(args)
            for i in range(iterations):
                # Load training data from filesystem
                data = np.zeros((batch_size,)+input_shape)
                labels = np.zeros((batch_size, 1))
                data_index = 0
                sample_pointer = sample_offset

                print()
                print('Iteration:', i, 'of', iterations)
                print('Time elapsed:', int(time.time())-now, 'sec')
                print('Loading', batch_size, 'samples starting at', sample_pointer)
                for i in range(batch_size):
                    class_id = samples[sample_pointer][1]
                    sample_path = samples[sample_pointer][2]
                    sample = get_mutations(sample_path, 1)[0]
                    sample = np.array(cv2.normalize(sample.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)).reshape(input_shape)
                    data[i] = sample
                    labels[i] = class_id

                    sample_pointer += 1
                    if sample_pointer>=len(samples): sample_pointer = 0

                sample_offset = sample_pointer
                if sample_offset>=len(samples): sample_offset = 0
        
                one_hot_labels = keras.utils.to_categorical(labels, num_classes=num_classes) # Convert labels to categorical one-hot encoding

                # Train the model
                if len(validation_data) and len(validation_labels):
                    print('Training with validation data...')
                    model.fit(data, one_hot_labels, batch_size=batch_size, epochs=epochs, verbose=0,
                        callbacks=[history], validation_data=(validation_data, validation_labels))
                else:
                    print('Training...')
                    model.fit(data, one_hot_labels, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[history])
                #model.train_on_batch(data, one_hot_labels)

                model.save_weights(weights_path)
    
def main():
    class_ids = list()
    srcdir = os.path.normpath(ARGS.src)
    pathlist = os.listdir(srcdir)
    for path in pathlist:
        class_ids.append(int(path))

    num_classes = len(class_ids)

    input_shape = (ARGS.dimension, ARGS.dimension, 1)

    model_dir = os.path.normpath(ARGS.model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model = get_model(input_shape, num_classes, model_dir, ARGS)

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
            predictions = model.predict(data)
            predictions = np.argmax(predictions, axis=1)
            predictions = np.column_stack((np.array(file_list), np.array(predictions), np.array(class_list)))
            for file_path, prediction, class_id in predictions:
                if prediction!=class_id:
                    errors += 1
                    print(file_path, prediction, class_id)
                    print('Error rate:', errors/total_count)
            file_list = list()
            class_list = list()
            data_index = 0

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Train the model""")
    parser.add_argument(
        '--src',
        type=str,
        default='../../data/GTSRB/processed/train',
        help='Path to source directory of training images.'
    )
    parser.add_argument(
        '--validation',
        type=str,
        default='../../data/GTSRB/processed/test',
        help='Path to source directory of validation images.'
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
        default='../../models/lenet',
        help='Path to directory of models and weights.'
    )
    parser.add_argument(
        '--model_prototype',
        type=str,
        default='ss',
        help='The name of model prototype to use with pre-defined hyperparameters.'
    )
    parser.add_argument(
        '--dimension',
        type=int,
        default=32,
        help='Target dimension of prepared samples.'
    )
    parser.add_argument(
        '--mintensity',
        type=float,
        default=1,
        help='The intensity of mutation.'
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