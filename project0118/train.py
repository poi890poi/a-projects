import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot

import argparse
import os.path
import os
from pathlib import Path
import random
import time
import json

import cv2
import numpy as np
import scipy.signal
import pydot_ng as pydot

from datagen import get_mutations, create_empty_directory
from uuid import uuid4

from keras.models import Model, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, Input, Embedding, concatenate
from keras.optimizers import Adam
import keras.callbacks
import keras.utils

print(keras.__version__)
print('initialized')

ARGS = None

class TrainHistory(keras.callbacks.Callback):
    def __init__(self, args, model_dir):
        super(keras.callbacks.Callback, self).__init__()
        self.sampling_window = 8
        self.sampling_count = self.sampling_window
        self.losses = list()
        self.acc = list()
        self.epoch = 1

        mptype = args.model_prototype
        self.graph_path = os.path.join(model_dir, 'history.png')

    def update_graph(self):
        try:
            losses = np.array(self.losses)
            acc = np.array(self.acc)

            fig, ax1 = matplotlib.pyplot.subplots()
            x_axis = np.array(range(len(losses))) * self.sampling_window
            ax1.plot(x_axis, self.losses, 'r:', linewidth=1.0)
            ax1.set_ylabel('Loss', color='r')
            ax1.set(xlabel='epochs',
                title='Training history')
            ax1.grid()

            ax2 = ax1.twinx()
            ax2.plot(x_axis, self.acc, 'g', linewidth=1.0)
            ax2.set_ylabel('Accuracy', color='g')

            fig.savefig(self.graph_path)
        except Exception:
            pass
        
        try:
            matplotlib.pyplot.close(fig)
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

        #print('epochs:', self.epoch, ', loss:', loss, ', acc:', acc)
        self.epoch += 1

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

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Hyperparameters(metaclass=Singleton):

    def __init__(self, prototype):
        self.prototype = prototype
        self.FLAG_BATCHNORMALIZATION = 1
        parameters = {
            'ss' : {
                'flags' : 0,
                'f' : [20, 50], # Number of feature maps for conv_1 and conv_2
                'p' : [2, 2], # Window size of pooling
                'fc' : [500, 0], # Size of full-connected layer
                'd' : [0, 0], # Dropout rate
                'lr' : 0.001, # Initial learn_rate
                'lr_ft' : 0.0001, # Fine-tune learn_rate
                'it' : 64, # Number of iterations
                'ft' : 48, # Fine-tune after N iterations
                'bs' : 4096, # batch_size
                'vs' : 0.2, # validation_split
                'ep' : 256, # epochs
                'es-md' : 0.0001, # min_delta for EarlyStopping
                'es-pt' : 16, # patientce (epochs) for EarlyStopping
                'met-es' : 'val_loss', # Monitoring metric for EarlyStopping
                'met-cp' : 'val_loss', # Monitoring metric for CheckPoint
            },
            'ms' : {
                'flags' : 1,
                'f' : [32, 64], # Number of feature maps for conv_1 and conv_2
                'p' : [2, 4], # Window size of pooling
                'fc' : [400, 400], # Size of full-connected layer
                'd' : [0.5, 0.5], # Dropout rate
                'lr' : 0.002, # Initial learn_rate
                'lr_ft' : 0.0001, # Fine-tune learn_rate
                'it' : 4096, # Number of iterations
                'ft' : 2048, # Fine-tune after N iterations
                'bs' : 512, # batch_size
                'vs' : 0.2, # validation_split
                'ep' : 256, # epochs
                'es-md' : 0.0001, # min_delta for EarlyStopping
                'es-pt' : 8, # patientce (epochs) for EarlyStopping
                'met-es' : 'val_loss', # Monitoring metric for EarlyStopping
                'met-cp' : 'val_loss', # Monitoring metric for CheckPoint
            },
            'bn' : {
                'flags' : 1,
                'f' : [32, 64], # Number of feature maps for conv_1 and conv_2
                'p' : [2, 4], # Window size of pooling
                'fc' : [400, 400], # Size of full-connected layer
                'd' : [0.5, 0.5], # Dropout rate
                'lr' : 0.002, # Initial learn_rate
                'lr_ft' : 0.0001, # Fine-tune learn_rate
                'it' : 4096, # Number of iterations
                'ft' : 2048, # Fine-tune after N iterations
                'bs' : 512, # batch_size
                'vs' : 0.2, # validation_split
                'ep' : 256, # epochs
                'es-md' : 0.0001, # min_delta for EarlyStopping
                'es-pt' : 16, # patientce (epochs) for EarlyStopping
                'met-es' : 'val_loss', # Monitoring metric for EarlyStopping
                'met-cp' : 'val_acc', # Monitoring metric for CheckPoint
            },
        }
        self.f1 = parameters[prototype]['f'][0]
        self.f2 = parameters[prototype]['f'][1]
        self.p1 = parameters[prototype]['p'][0]
        self.p2 = parameters[prototype]['p'][1]
        self.fc1 = parameters[prototype]['fc'][0]
        self.fc2 = parameters[prototype]['fc'][1]
        self.d1 = parameters[prototype]['d'][0]
        self.d2 = parameters[prototype]['d'][1]
        self.lr = parameters[prototype]['lr']
        self.flags = parameters[prototype]['flags']
        self.iterations = parameters[prototype]['it']
        self.iterations_ft = parameters[prototype]['ft']
        self.batch_size = parameters[prototype]['bs']
        self.validation_size = parameters[prototype]['vs']
        self.epochs = parameters[prototype]['ep']
        self.learn_rate = parameters[prototype]['lr']
        self.lr_fine_tune = parameters[prototype]['lr_ft']
        self.min_delta = parameters[prototype]['es-md']
        self.patience = parameters[prototype]['es-pt']
        self.earlystop_metric = parameters[prototype]['met-es']
        self.checkpoint_metric = parameters[prototype]['met-cp']
        
    def dump(self)
        return json.dump(parameters[self.prototype])

    def use_batch_norm(self):
        return self.flags&self.FLAG_BATCHNORMALIZATION

    def use_multi_scale(self):
        return self.fc2

    def early_stop(self):
        return self.min_delta&self.patience

def get_model(input_shape, num_classes, model_dir, args):
    hp = Hyperparameters(args.model_prototype)

    model_path = os.path.join(model_dir, 'model.json')
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
        x = Conv2D(hp.f1, (5, 5), kernel_initializer='glorot_normal', input_shape=input_shape, name='conv_1')(in_raw)
        x = BatchNormalization()(x)
        x = Activation('relu', name='relu_1')(x)
        x = MaxPooling2D(pool_size=(hp.p1, hp.p1), name='maxpool_1')(x)

        # Define output of stage-1
        in_s1 = Flatten()(x)

        # Begin of stage-2
        x = Conv2D(hp.f2, (5, 5), kernel_initializer='glorot_normal', input_shape=input_shape, name='conv_2')(x)
        if (hp.use_multi_scale()): x = BatchNormalization()(x)
        x = Activation('relu', name='relu_2')(x)
        x = MaxPooling2D(pool_size=(hp.p2, hp.p2), name='maxpool_2')(x)
        x = Flatten()(x)

        # Concatenate outputs from stage-1 and stage-2
        if (hp.use_multi_scale()): x = concatenate([x, in_s1])

        # 1st fully-connected layer
        x = Dense(hp.fc1, name='fc_1')(x)
        if (hp.use_multi_scale()): x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if hp.d1: x = Dropout(hp.d1)(x)

        # 2nd (optional) fully-connected layer
        if hp.fc2:
            x = Dense(hp.fc2, name='fc_2')(x)
            if (hp.use_multi_scale()): x = BatchNormalization()(x)
            x = Activation('relu')(x)
            if hp.d2: x = Dropout(hp.d2)(x)

        x = Dense(num_classes)(x)
        if (hp.use_multi_scale()): x = BatchNormalization()(x)
        predictions = Activation('softmax', name='softmax')(x)

        model = Model(inputs=in_raw, outputs=predictions)

        model_json = model.to_json()
        with open(model_path, 'w') as model_file:
            model_file.write(model_json)

        hp_path = os.path.join(model_dir, 'parameters.json')
        with open(hp_path, 'w') as f:
            f.write(Hyperparameters(args.model_prototype).dump())

    # Save model plot
    graph_path = os.path.join(model_dir, 'model.png')
    keras.utils.plot_model(model, to_file=graph_path)

    adam = Adam(lr=lr)
    model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_or_load(model, input_shape, class_ids, model_dir, args):
    hp = Hyperparameters(args.model_prototype)

    num_classes = len(class_ids)

    weights_path = os.path.join(model_dir, 'weights.keras')
    if os.path.exists(weights_path) and not args.reset:
        # Load weights from file
        model.load_weights(weights_path)
        print()
        print('Weights loaded from', weights_path)

    if args.dummy:
        # Generate random dummy data for verification of model definition
        data = np.random.random((1000,)+input_shape)
        labels = np.random.randint(num_classes, size=(1000, 1))
        one_hot_labels = keras.utils.to_categorical(labels, num_classes=num_classes) # Convert labels to categorical one-hot encoding
        model.fit(data, one_hot_labels, batch_size=32, epochs=epochs, verbose=1)
    else:
        # Get list of proto-samples and shuffle them
        print()
        print('Getting sample file list...')
        sample_count = 0
        now = int(time.time())
        proto_samples = list() # List of all proto-samples (in the original dataset without mutation) sorted by classes
        for i in range(num_classes):
            proto_samples.append(list())
        srcdir = os.path.normpath(args.src)
        pathlist = sorted(Path(srcdir).glob('**/*.ppm'))
        for path in pathlist:
            path_in_str = str(path)
            sample_dir, sample_filename = os.path.split(path_in_str)
            class_id = int(os.path.split(sample_dir)[1])
            proto_samples[class_id].append(path_in_str)
            sample_count += 1
        for i in range(num_classes):
            random.shuffle(proto_samples[i])
        print('Total of', sample_count, 'samples found.')

        # These variables/objects outside of iteration loop maintain their status through the whole training process
        sample_offset = 0
        history = TrainHistory(args, model_dir)
        logdir = os.path.normpath(os.path.join(model_dir, 'logdir'))
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        
        # The graph is inconsistent with keras
        #tensorboard = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=epochs/4,
        #    write_graph=True, write_grads=True, write_images=True)

        # Iterations of training
        samples_per_class = int(hp.batch_size / num_classes)
        hp.batch_size = int(num_classes * samples_per_class) # Enforce strict classes balancing (same amount of samples for each class)
        sample_offset = 0
        fine_tune = False
        for iteration in range(hp.iterations):
            # Load training data from filesystem
            samples = list()

            if iteration >= int(hp.iterations_ft):
                fine_tune = True
            print()
            if fine_tune:
                print('Fine-tuning iteration:', iteration, 'in', hp.iterations)
            else:
                print('Initial iteration:', iteration, 'in', hp.iterations)
            print('Time elapsed:', int(time.time())-now, 'sec')
            print('Loading', samples_per_class, 'samples per class with offset', sample_offset)
            for i in range(samples_per_class):
                batch_shuffled = list()
                for class_id in range(num_classes):
                    sample_pointer = (sample_offset + i) % len(proto_samples[class_id])
                    #print(class_id, len(samples[class_id]), sample_pointer)
                    #print(samples[class_id])
                    sample_path = proto_samples[class_id][sample_pointer]
                    if fine_tune:
                        sample = get_mutations(sample_path, 1, intensity=0.75)[0]
                    else:
                        sample = get_mutations(sample_path, 1, intensity=1.0)[0]
                    sample = np.array(cv2.normalize(sample.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)).reshape(input_shape)
                    batch_shuffled.append([class_id, sample]) # Pair class_id and sample data so it's easily shuffled
                random.shuffle(batch_shuffled) # Shuffle in small chunk with balanced data (one sample for each class)
                samples += batch_shuffled

            labels = np.zeros((len(samples), 1))
            data = np.zeros((len(samples),)+input_shape)
            for i in range(len(samples)):
                labels[i] = samples[i][0]
                data[i] = samples[i][1]
            one_hot_labels = keras.utils.to_categorical(labels, num_classes=num_classes) # Convert labels to categorical one-hot encoding

            # Train the model
            callbacks = [
                history,
                #tensorboard, # The graph is messed and inconsistent with keras
                keras.callbacks.ModelCheckpoint(weights_path, monitor=hp.checkpoint_metric, save_best_only=True, verbose=0),
            ]
            if hp.min_delta&hp.patience:
                callbacks.append(keras.callbacks.EarlyStopping(monitor=hp.earlystop_metric, min_delta=hp.min_delta, patience=hp.patience, verbose=1))

            if fine_tune:
                print('Fine-tuning with learn rate=', hp.lr_fine_tune)
                model.optimizer.lr.assign(hp.lr_fine_tune)
            else:
                print('Initial training with learn rate=', hp.learn_rate)
                model.optimizer.lr.assign(hp.learn_rate)
            model.fit(data, one_hot_labels,
                batch_size=hp.batch_size, epochs=hp.epochs,
                verbose=1, callbacks=callbacks,
                validation_split=0.2, shuffle=False) # Do not use keras internal shuffling so the logic can be tweaked
            #model.train_on_batch(data, one_hot_labels)

            sample_offset += samples_per_class
            if sample_offset>=65535: sample_offset = 0

        model.save_weights(weights_path)
    
def main():
    class_ids = list()
    srcdir = os.path.normpath(ARGS.src)
    pathlist = os.listdir(srcdir)
    for path in pathlist:
        class_ids.append(int(path))

    num_classes = len(class_ids)

    input_shape = (ARGS.dimension, ARGS.dimension, 1)

    model_dir = os.path.normpath(os.path.join(ARGS.model_dir, ARGS.model_prototype))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model = get_model(input_shape, num_classes, model_dir, ARGS)

    train_or_load(model, input_shape, class_ids, model_dir, ARGS)

    print()
    print('done')

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