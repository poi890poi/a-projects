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
    """model_dir = '../models/' + args.model
    tf.reset_default_graph()
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], model_dir)"""

    prep_denoise = False
    prep_equalize = True

    time_start = time.time()
    model = SimpleClassifier(args.model, args.train_id)
    classifier = model.get_estimator()
    time_diff = time.time() - time_start
    print('tf.estimator created, dt:', time_diff)

    data_dir = '../data/face/lfw/extracted/lfw'
    count = 256
    while count:
        time_start = time.time()
        f = DirectoryWalker().get_a_file(directory=data_dir, filters=['.jpg'])
        image = cv2.imread(f.path, 1)
        time_diff = time.time() - time_start
        print('image loaded, dt:', time_diff)

        time_start = time.time()
        image = ImageUtilities.preprocess(image, convert_gray=None, equalize=prep_equalize, denoise=prep_denoise, maxsize=500)
        #image = np.array(image, dtype=np.float32)/255
        time_diff = time.time() - time_start
        print('image processed, dt:', time_diff)

        time_start = time.time()
        src_shape = image.shape
        gray = ImageUtilities.preprocess(image, convert_gray=cv2.COLOR_RGB2YCrCb, equalize=prep_equalize, denoise=prep_denoise)
        processed_shape = gray.shape
        mrate = [processed_shape[0]/src_shape[0], processed_shape[1]/src_shape[1]]
        time_diff = time.time() - time_start
        print('image converted to gray, dt:', time_diff)

        time_start = time.time()
        rects, landmarks = FaceDetector().detect(gray)
        time_diff = time.time() - time_start
        print('detected with svm, dt:', time_diff)
        face = None
        for rect in rects:
            (x, y, w, h) = ImageUtilities.rect_to_bb(rect, mrate=mrate)
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            height, width, *rest = image.shape
            (x, y, w, h) = ImageUtilities.rect_fit_ar(np.array([x, y, w, h], dtype=np.int), [0, 0, width, height], 2/3, mrate=1.25)
            if w<=0 and h<=0:
                continue

            face = np.array(image[y:y+h, x:x+w, :])
            face = ImageUtilities.preprocess(face, convert_gray=None)
            face = imresize(face, [96, 64])
            data = (np.array(np.copy(face), dtype=np.float32)/255).reshape((1,)+face.shape)
            time_start = time.time()
            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'x': data},
                batch_size=1,
                shuffle=False)
            predictions = classifier.predict(
                predict_input_fn
            )
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            for p in predictions:
                c = p['classes']
                print('class:', c, ', confidence:', p['probabilities'][c])
                if c==1:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            time_diff = time.time() - time_start
            print('predicetd with cnn, dt:', time_diff)


        canvas = ViewportManager().open('preview', shape=image.shape, blocks=(1, 2))
        ViewportManager().put('preview', image, (0, 0))
        if face is not None: ViewportManager().put('preview', face, (0, 1))
        ViewportManager().update('preview')

        k = ViewportManager().wait_key()
        if k in (ViewportManager.KEY_ENTER, ViewportManager.KEY_SPACE):
            pass
        print()
        print('next')
        print()
