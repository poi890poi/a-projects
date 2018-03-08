import cv2
import numpy as np
import dlib
from scipy.misc import imresize

from shared.utilities import DataUtilities, ImageUtilities, DirectoryWalker, ViewportManager

import os
import cProfile, pstats
import linecache, tracemalloc
import psutil


def memory_usage_psutil(checkpoint):
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    print()
    print(checkpoint, mem)
    return mem

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class FaceDetector(metaclass=Singleton):
    def __init__(self):
        self.detector = None
        self.predictor = None

    @staticmethod
    def landmarks2coords(marks, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)
    
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (marks.part(i).x, marks.part(i).y)
    
        # return the list of (x, y)-coordinates
        return coords

    def detect(self, img, request_landmarks=False):
        if self.detector is None: self.detector = dlib.get_frontal_face_detector()
        if self.predictor is None: self.predictor = dlib.shape_predictor('./face/pretrained/dlib/shape_predictor_68_face_landmarks.dat')

        rects = self.detector(img, 1)
        landmarks = list()

        for (i, rect) in enumerate(rects):
            if request_landmarks:
                lm = self.predictor(img, rect)
                lm = landmarks2coords(lm)
                landmarks.append(lm)

        return (rects, landmarks)

def detect(inpath):
    image = cv2.imread(inpath, 1)
    gray = ImageUtilities.preprocess(image, convert_gray=cv2.COLOR_RGB2YCrCb, maxsize=640)
    height, width, *rest = gray.shape
    image = imresize(image, [height, width])
    memory_usage_psutil('Image loaded')

    canvas = ViewportManager().open('preview', shape=image.shape, blocks=(1, 1))
    memory_usage_psutil('Preview window opened')

    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        (x, y, w, h) = ImageUtilities.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        landmarks = predictor(gray, rect)
        landmarks = landmarks2coords(landmarks)
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
 
    ViewportManager().put('preview', image, (0,0))
    ViewportManager().update('preview')


def main():
    memory_usage_psutil('Initial')

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./face/pretrained/dlib/shape_predictor_68_face_landmarks.dat')

    memory_usage_psutil('SVM loaded')

    while True:

        f = DirectoryWalker().get_a_file(directory='../data/face/wiki-face/extracted/wiki', filters=['.jpg'])
        if f is None:
            print('No more sample T_T')
            break

        detect(f.path)
        
        continue

        cProfile.run('detect(f.path)', 'restats')
        p = pstats.Stats('restats')
        p.sort_stats('tottime')
        print()
        p.print_stats(10)

        k = ViewportManager().wait_key()
        if k in (ViewportManager.KEY_ENTER, ViewportManager.KEY_SPACE):
            pass

