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
from keras import backend as K

print(keras.__version__)
print('initialized')

ARGS = None

class TrainHistory(keras.callbacks.Callback):
    def __init__(self, args, model_dir):
        super(keras.callbacks.Callback, self).__init__()
        self.sampling_window = 16
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
        loss = logs.get('val_loss')
        acc = logs.get('val_acc')
        if self.sampling_count>=self.sampling_window:
            # Expand history list
            self.losses.append(loss)
            self.acc.append(acc)
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
        self.parameters = {
            'ss' : {
                'flags' : 0,
                'k' : [5, 5], # Kernel size for conv_1 and conv_2
                'f' : [20, 50, 0], # Number of feature maps for conv_1 and conv_2
                'p' : [2, 2, 4], # Window size of pooling
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
                'k' : [5, 5], # Kernel size for conv_1 and conv_2
                'f' : [32, 64], # Number of feature maps for conv_1 and conv_2
                'p' : [2, 4, 4], # Window size of pooling
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
                'k' : [5, 5], # Kernel size for conv_1 and conv_2
                'f' : [32, 64, 0], # Number of feature maps for conv_1 and conv_2
                'p' : [2, 2, 4], # Window size of pooling
                'fc' : [400, 400], # Size of full-connected layer
                'd' : [0.5, 0.5], # Dropout rate
                'lr' : 0.001, # Initial learn_rate
                'lr_ft' : 0.00001, # Fine-tune learn_rate
                'it' : 512, # Number of iterations
                'ft' : 256, # Fine-tune after N iterations
                'bs' : 1024, # batch_size
                'vs' : 0.4, # validation_split
                'ep' : 256, # epochs
                'es-md' : 0.0001, # min_delta for EarlyStopping
                'es-pt' : 16, # patientce (epochs) for EarlyStopping
                'met-es' : 'val_loss', # Monitoring metric for EarlyStopping
                'met-cp' : 'val_acc', # Monitoring metric for CheckPoint
            },
            'as' : { # Alex Staravoitau https://navoshta.com/traffic-signs-classification/
                'flags' : 1,
                'k' : [5, 5, 5], # Kernel size for conv_1 and conv_2
                'f' : [32, 64, 128], # Number of feature maps for conv_1 and conv_2
                'p' : [2, 2, 4, 2, 2], # Window size of pooling
                'fc' : [1024, 0], # Size of full-connected layer
                'd' : [0.5, 0, 0.1, 0.2, 0.3], # Dropout rate
                'it' : [32, 32, 32, 64, 128], # Number of iterations for each phase
                'lr' : [0.001, 0.0005, 0.0001, 0.00002, 0.00001], # Learn rate for each phase
                'mi' : [1.0, 0.9, 0.85, 0.75], # Mutation intensity for each phase
                'bs' : 512, # batch_size
                'vs' : 0.2, # validation_split
                'ep' : 256, # epochs
                'es-md' : 0.0001, # min_delta for EarlyStopping
                'es-pt' : 16, # patientce (epochs) for EarlyStopping
                'met-es' : 'val_loss', # Monitoring metric for EarlyStopping
                'met-cp' : 'val_acc', # Monitoring metric for CheckPoint
            },
        }
        self.k1 = self.parameters[prototype]['k'][0]
        self.k2 = self.parameters[prototype]['k'][1]
        self.k3 = self.parameters[prototype]['k'][2]
        self.f1 = self.parameters[prototype]['f'][0]
        self.f2 = self.parameters[prototype]['f'][1]
        self.f3 = self.parameters[prototype]['f'][2]
        self.p1 = self.parameters[prototype]['p'][0]
        self.p2 = self.parameters[prototype]['p'][1]
        self.pi1 = self.parameters[prototype]['p'][2]
        self.pi2 = self.parameters[prototype]['p'][3]
        self.p3 = self.parameters[prototype]['p'][4]
        self.fc1 = self.parameters[prototype]['fc'][0]
        self.fc2 = self.parameters[prototype]['fc'][1]
        self.d1 = self.parameters[prototype]['d'][0]
        self.d2 = self.parameters[prototype]['d'][1]
        self.dc1 = self.parameters[prototype]['d'][2]
        self.dc2 = self.parameters[prototype]['d'][3]
        self.dc3 = self.parameters[prototype]['d'][4]
        self.vs = self.parameters[prototype]['vs']
        self.flags = self.parameters[prototype]['flags']
        self.iterations = self.parameters[prototype]['it']
        iterations = list(self.iterations)
        for i in range(len(self.iterations)):
            iterations[i] = int(np.sum(self.iterations[:i+1])) # numpy integer must be casted to int for Python to JSON-encode
            print(iterations[i])
        self.iterations = list(iterations)
        self.iterations_total = self.iterations[-1]
        self.learn_rate = self.parameters[prototype]['lr']
        self.mintensity = self.parameters[prototype]['mi']
        self.batch_size = self.parameters[prototype]['bs']
        self.validation_size = self.parameters[prototype]['vs']
        self.epochs = self.parameters[prototype]['ep']
        self.min_delta = self.parameters[prototype]['es-md']
        self.patience = self.parameters[prototype]['es-pt']
        self.earlystop_metric = self.parameters[prototype]['met-es']
        self.checkpoint_metric = self.parameters[prototype]['met-cp']
        
    def dump(self):
        return json.dumps(self.parameters[self.prototype])

    def use_batch_norm(self):
        return self.flags&self.FLAG_BATCHNORMALIZATION

    def use_multi_scale(self):
        return self.fc2

    def early_stop(self):
        return self.min_delta&self.patience

def get_model(input_shape, num_classes, model_dir, args):
    hp = Hyperparameters(args.model_prototype)

    print()
    model_path = os.path.join(model_dir, 'model.json')
    model = None
    if os.path.exists(model_path) and not args.reset:
        # Load model from file
        with open(model_path, 'r') as model_file:
            model_json = model_file.read()
            model = model_from_json(model_json)
        print('Model loaded from', model_path)
    else:
        # Define LeNet multi-scale model
        in_raw = Input(shape=input_shape) # Raw images as source input

        x = Conv2D(hp.f1, (hp.k1, hp.k1), kernel_initializer='glorot_normal', input_shape=input_shape, padding='same', name='conv_1')(in_raw)
        if (hp.use_batch_norm()): x = BatchNormalization()(x)
        x = Activation('relu', name='relu_1')(x)
        x = MaxPooling2D(pool_size=(hp.p1, hp.p1), name='maxpool_1')(x)
        if hp.dc1: x = Dropout(hp.dc1)(x)

        # Define output of stage-1
        if hp.pi1: in_s1 = MaxPooling2D(pool_size=(hp.pi1, hp.pi1))(x)

        # Begin of stage-2
        if hp.f2:
            x = Conv2D(hp.f2, (hp.k2, hp.k2), kernel_initializer='glorot_normal', input_shape=input_shape, padding='same', name='conv_2')(x)
            print('conv_2', x)
            if (hp.use_batch_norm()): x = BatchNormalization()(x)
            x = Activation('relu', name='relu_2')(x)
            x = MaxPooling2D(pool_size=(hp.p2, hp.p2), name='maxpool_2')(x)
            if hp.dc2: x = Dropout(hp.dc2)(x)

            # Define output of stage-1
            if hp.pi2: in_s2 = MaxPooling2D(pool_size=(hp.pi2, hp.pi2))(x)

        # Begin of stage-3
        if hp.f3:
            x = Conv2D(hp.f3, (hp.k3, hp.k3), kernel_initializer='glorot_normal', input_shape=input_shape, padding='same', name='conv_3')(x)
            print('conv_3', x)
            if (hp.use_batch_norm()): x = BatchNormalization()(x)
            x = Activation('relu', name='relu_3')(x)
            x = MaxPooling2D(pool_size=(hp.p3, hp.p3), name='maxpool_3')(x)
            if hp.dc3: x = Dropout(hp.dc3)(x)

        # Concatenate outputs from stage-1, stage-2, and stage-3
        x = Flatten()(x)
        if hp.pi1:
            conc = [x]
            print('in_s1', in_s1)
            in_s1 = Flatten()(in_s1)
            conc.append(in_s1)
            if hp.pi2:
                print('in_s2', in_s2)
                in_s2 = Flatten()(in_s2)
                conc.append(in_s2)
            x = concatenate(conc)
        
        print('concatenated', x)

        # 1st fully-connected layer
        x = Dense(hp.fc1, name='fc_1')(x)
        print('fc_1', x)
        if (hp.use_multi_scale()): x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if hp.d1: x = Dropout(hp.d1)(x)

        # 2nd (optional) fully-connected layer
        if hp.fc2:
            x = Dense(hp.fc2, name='fc_2')(x)
            print('fc_2', x)
            if (hp.use_batch_norm()): x = BatchNormalization()(x)
            x = Activation('relu')(x)
            if hp.d2: x = Dropout(hp.d2)(x)

        print('softmax in', x)
        x = Dense(num_classes)(x)
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

    adam = Adam(lr=hp.learn_rate[0])
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
        for iteration in range(hp.iterations_total):
            # Load training data from filesystem
            samples = list()

            phase = 0
            for i in range(len(hp.iterations)-1):
                if iteration > hp.iterations[i]:
                    phase = i + 1
            print()
            print('Initial iteration:', iteration, 'in', hp.iterations_total, ', phase', phase)
            print('Time elapsed:', int(time.time())-now, 'sec')
            print('Loading', samples_per_class, 'samples per class with offset', sample_offset)
            for i in range(samples_per_class):
                batch_shuffled = list()
                for class_id in range(num_classes):
                    sample_pointer = (sample_offset + i) % len(proto_samples[class_id])
                    #print(class_id, len(samples[class_id]), sample_pointer)
                    #print(samples[class_id])
                    sample_path = proto_samples[class_id][sample_pointer]
                    sample = get_mutations(sample_path, 1, intensity=hp.mintensity[phase])[0]
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
                keras.callbacks.ModelCheckpoint(weights_path, monitor=hp.checkpoint_metric, save_best_only=True, verbose=1),
            ]
            if hp.patience:
                callbacks.append(keras.callbacks.EarlyStopping(monitor=hp.earlystop_metric, min_delta=hp.min_delta, patience=hp.patience, verbose=1))

            lr = hp.learn_rate[phase]
            print('Training with learn rate=', lr)
            K.update(model.optimizer.lr, lr)
            model.fit(data, one_hot_labels,
                batch_size=num_classes, epochs=hp.epochs,
                verbose=1, callbacks=callbacks,
                validation_split=hp.vs, shuffle=False) # Do not use keras internal shuffling so the logic can be controlled
            #model.train_on_batch(data, one_hot_labels)

            sample_offset += samples_per_class
            if sample_offset>=65535: sample_offset = 0
    
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
        default='bn',
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