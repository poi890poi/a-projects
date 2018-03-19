import time
from scipy.misc import imresize
import numpy as np
import math

import dlib

import face_recognition

predictor_path = '../../lib/dlib/shape_predictor_5_face_landmarks.dat'
face_rec_model_path = '../../lib/dlib/dlib_face_recognition_resnet_model_v1.dat'

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

def fit_pixel_count(img, maxsize=[640, 480]):
    height, width, *_ = img.shape
    a = height*width
    a_ = maxsize[0]*maxsize[1]
    rate = 1.
    if a > a_:
        rate = math.sqrt(a_/a)
    img = imresize(img, (np.array([height, width], dtype=np.float)*rate).astype(np.int))
    return (img, rate)

def fit_resize(img, maxsize=[320, 200], inner_fit=False):
    height, width, *_ = img.shape
    if inner_fit:
        rate = min(maxsize[1]/height, maxsize[0]/width)
    else:
        rate = max(maxsize[1]/height, maxsize[0]/width)
    img = imresize(img, (np.array([height, width], dtype=np.float)*rate).astype(np.int))
    return (img, rate)

time_start = time.time()
image = face_recognition.load_image_file('../../data/face/demo/positive/2690900_1963-03-12_2010.jpg')
time_diff = time.time() - time_start
print('load image', time_diff)

print('original size', image.shape)
image, rate = fit_pixel_count(image, [640, 480])
print('resized', image.shape, rate)

time_start = time.time()
for i in range(100):
    face_locations = detector(image, 1)

time_diff = time.time() - time_start
print('get_frontal_face_detector', time_diff/100*1000, 'ms')
print(len(face_locations), face_locations)

time_start = time.time()
for i in range(100):
    face_locations = face_recognition.face_locations(image)
time_diff = time.time() - time_start
print('face_recognition.face_locations', time_diff/100*1000, 'ms')
print(len(face_locations), face_locations)

image = face_recognition.load_image_file('../../data/face/val/positive/wiki/ffca946fedd0f12c8ec941e7cef86c72df6f6d80.jpg')
image, rate = fit_resize(image, [12, 12])

"""time_start = time.time()
for i in range(100):
    face_descriptor = facerec.compute_face_descriptor(image, image.shape)
time_diff = time.time() - time_start
print('facerec.compute_face_descriptor', time_diff/100*1000, 'ms')
#print(encodings)"""

time_start = time.time()
for i in range(100):
    encodings = face_recognition.face_encodings(image)
time_diff = time.time() - time_start
print('face_recognition.face_encodings', time_diff/100*1000, 'ms')
print(len(encodings))