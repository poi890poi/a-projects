from __future__ import division, absolute_import
import re
import numpy as np
from emotion.dataset_loader import DatasetLoader
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from emotion.constants import *
from os.path import isfile, join
import random
import sys

import cv2
from shared.utilities import DirectoryWalker
import os.path

class EmotionRecognition:

  def __init__(self):
    self.dataset = DatasetLoader()

  def build_network(self):
    # Smaller 'AlexNet'
    # https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py
    print('[+] Building CNN')
    self.network = input_data(shape = [None, SIZE_FACE, SIZE_FACE, 1])
    self.network = conv_2d(self.network, 64, 5, activation = 'relu')
    #self.network = local_response_normalization(self.network)
    self.network = max_pool_2d(self.network, 3, strides = 2)
    self.network = conv_2d(self.network, 64, 5, activation = 'relu')
    self.network = max_pool_2d(self.network, 3, strides = 2)
    self.network = conv_2d(self.network, 128, 4, activation = 'relu')
    self.network = dropout(self.network, 0.3)
    self.network = fully_connected(self.network, 3072, activation = 'relu')
    self.network = fully_connected(self.network, len(EMOTIONS), activation = 'softmax')
    self.network = regression(self.network,
      optimizer = 'momentum',
      loss = 'categorical_crossentropy')

    checkpoint_path = SAVE_DIRECTORY + '/emotion_recognition'
    print('checkpoint_path', checkpoint_path)
    self.model = tflearn.DNN(
      self.network,
      checkpoint_path = checkpoint_path,
      max_checkpoints = 1,
      tensorboard_verbose = 2
    )
    self.load_model()

  def load_saved_dataset(self):
    self.dataset.load_from_save()
    print('[+] Dataset found and loaded')

  def start_training(self):
    self.build_network()
    """self.load_saved_dataset()
    if self.dataset is None:
      self.load_saved_dataset()"""

    # Load training data
    data_sources = [
      ['../data/face/fer2013/privatetest', 'val'],
      ['../data/face/fer2013/training', 'train'],
    ]
      
    train_data = list()
    train_labels = list()
    val_data = list()
    val_labels = list()
    for source in data_sources:
      while True:
          f = DirectoryWalker().get_a_file(directory=source[0], filters=['.jpg'])
          if f is None: break
          img = cv2.imread(f.path, 0)
          if img is None:break

          components = os.path.split(f.path)
          label = int(os.path.split(components[0])[1])
          components = os.path.split(components[0])
          subset = os.path.split(components[0])[1]
          #print(subset, label)

          img = (img.astype(dtype=np.float))/255

          if subset=='privatetest':
            val_data.append(img)
            val_labels.append(label)
          elif subset=='training':
            train_data.append(img)
            train_labels.append(label)

    train_data = np.array(train_data).reshape((-1, 48, 48, 1))
    onehot = np.zeros((len(train_labels), 7))
    onehot[np.arange(len(train_labels)), train_labels] = 1
    train_labels = onehot
    val_data = np.array(val_data).reshape((-1, 48, 48, 1))
    onehot = np.zeros((len(val_labels), 7))
    onehot[np.arange(len(val_labels)), val_labels] = 1
    val_labels = onehot

    # Training
    print('[+] Training network')
    self.model.fit(
      train_data, train_labels,
      validation_set = (val_data, val_labels),
      n_epoch = 100,
      batch_size = 50,
      shuffle = True,
      show_metric = True,
      snapshot_step = 200,
      snapshot_epoch = True,
      run_id = 'emotion_recognition'
    )

  def predict(self, image):
    image = image.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    return self.model.predict(image)

  def save_model(self):
    self.model.save(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
    print('[+] Model trained and saved at ' + SAVE_MODEL_FILENAME)

  def load_model(self):
    if isfile(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME)):
      self.model.load(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
      print('[+] Model loaded from ' + SAVE_MODEL_FILENAME)
    else:
      print('Model not found!', SAVE_DIRECTORY, SAVE_MODEL_FILENAME)


def show_usage():
  # I din't want to have more dependecies
  print('[!] Usage: python emotion_recognition.py')
  print('\t emotion_recognition.py train \t Trains and saves model with saved dataset')
  print('\t emotion_recognition.py poc \t Launch the proof of concept')

def fxpress(args):
  network = EmotionRecognition()
  network.build_network()
  inpath = '../data/face/fer2013/training/06/0b3d10353f329931ffef938e127ae77c694f1405.jpg'

  img = cv2.imread(inpath, 1)
  if img is None:
    print('Error reading image:', inpath)
    return
  print('input', img.shape)

  def format_image(image):
    print('format_image')
    if len(image.shape) > 2 and image.shape[2] == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
      image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    try:
      image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation = cv2.INTER_CUBIC) / 255.
    except Exception:
      print("[+] Problem during resize")
      return None
    print('shape', image.shape)
    return image

  result = network.predict(format_image(img))
  return result

def fxpress_train(args):
  network = EmotionRecognition()
  network.start_training()
  network.save_model()

if __name__ == "__main__":
  if len(sys.argv) <= 1:
    show_usage()
    exit()

  network = EmotionRecognition()
  if sys.argv[1] == 'train':
    network.start_training()
    network.save_model()
  elif sys.argv[1] == 'poc':
    import poc
  else:
    show_usage()
