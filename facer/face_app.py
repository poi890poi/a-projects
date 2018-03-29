from mtcnn import detect_face as FaceDetector
from facer.emotion import EmotionClassifier
from shared.utilities import ImageUtilities as imutil

import time
import numpy as np
import cv2
import tensorflow as tf

"""
- Array plane is (custom) defined as coordinate system of y (row) before x (col),
  optimizing for array and bytes operations
- Image plane is (custom) defined as coordinate system of x (col) before y (row),
  optimizing for image operations, specifically OpenCV compatibility
"""

class Rectangle:
    """
    Rectangle is (custom) defined with upper-left point (y, x) and length of 2 sides (h, w)
    """
    def __init__(self, y, x, h, w):
        self.rectangle = np.array([y, x, h, w], dtype=np.float)

    def to_extent(self):
        return Extent(self.rectangle[0], self.rectangle[1], self.rectangle[0]+self.rectangle[2], self.rectangle[1]+self.rectangle[3])

    def eval(self):
        return self.rectangle

class Extent:
    """
    Extent is (custom) defined with upper-left point (y1, x1) and lower-right point (y2, x2)
    """
    def __init__(self, y1, x1, y2, x2):
        self.extent = np.array([y1, x1, y2, x2], dtype=np.float)

    def to_rectangle(self):
        return Rectangle(self.extent[0], self.extent[1], self.extent[2]-self.extent[0], self.extent[3]-self.extent[1])

    def eval(self):
        return self.extent

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class VisionApplications(metaclass=Singleton):
    def detect(self, media): raise(NotImplementedError)

class FaceApplications(VisionApplications):
    def __init__(self):
        self.__mtcnn = None
        self.__emoc = None

    def __get_detector_create(self):
        if self.__mtcnn is None:
            self.__mtcnn = {}
            self.__mtcnn['session'] = tf.Session()
            #self.__pnet, self.__rnet, self.__onet = FaceDetector.create_mtcnn(self.__mtcnn, None)
            self.__mtcnn['pnet'], self.__mtcnn['rnet'], self.__mtcnn['onet'] = FaceDetector.create_mtcnn(self.__mtcnn['session'], None)
        return self.__mtcnn

    def __get_emotion_classifier_create(self):
        if self.__emoc is None:
            self.__emoc = EmotionClassifier()
            self.__emoc.build_network(None)
        return self.__emoc

    def detect(self, img, params):
        """
        Input: pixel array, shape (h, w)
        Ouput: list of rectangles of objects, shape (count, y, x, h, w)
        """
        predictions = {
            'rectangles': [],
            'confidences': [],
            'landmarks': [],
            'emotions': [],
            'timing': {},
        }
        if 'service' in params:
            #{'type': 'face', 'options': {'resultsLimit': 5}, 'model': '12-net'}
            service = params['service']
            model = service['model']

            # Limit image size for performance
            print('service', service)
            t_ = time.time()
            res_cap = 384
            factor = 0.709
            if 'options' in service:
                options = service['options']
                if 'res_cap' in options:
                    res_cap = int(options['res_cap'])
                if 'factor' in options:
                    factor = float(options['factor'])
            if res_cap > 512: res_cap = 512
            print('factor', factor)
            resized, scale_factor = imutil.fit_resize(img, maxsize=(res_cap, res_cap))
            scale_factor = 1. / scale_factor
            predictions['timing']['fit_resize'] = (time.time() - t_) * 1000

            print('mean', np.mean(img), np.mean(resized))

            #time_start = time.time()
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # MTCNN face detection
            t_ = time.time()
            detector = self.__get_detector_create()
            pnet = detector['pnet']
            rnet = detector['rnet']
            onet = detector['onet']
            extents, landmarks = FaceDetector.detect_face(resized, 40, pnet, rnet, onet, threshold=[0.6, 0.7, 0.9], factor=factor)
            predictions['timing']['mtcnn'] = (time.time() - t_) * 1000

            if len(extents):
                _ = np.array(landmarks) * scale_factor
                _ = _.reshape(2, -1)
                _ = np.transpose(_)
                _ = _.reshape((-1, len(extents), 2))
                landmarks = np.swapaxes(_, 0, 1)

            if model=='alexnet-emoc':
                facelist = np.zeros((len(extents), 48, 48), dtype=np.float)
                predictions['timing']['emoc_prepare'] = 0
            
            for i, e in enumerate(extents):
                #e_ = (Rectangle(r_[0], r_[1], r_[2], r_[3]).to_extent().eval() * scale_factor).astype(dtype=np.int).tolist()
                r_ = (np.array([e[0], e[1], e[2]-e[0], e[3]-e[1]])*scale_factor).astype(dtype=np.int).tolist()
                predictions['rectangles'].append(r_)
                predictions['confidences'].append(int(e[4]*1000))
                plist_ = landmarks[i].astype(dtype=np.int).tolist()
                predictions['landmarks'].append(plist_)
                if model=='alexnet-emoc':
                    t_ = time.time()
                    #(x, y, w, h) = imutil.rect_fit_ar(r_, [0, 0, img.shape[1], img.shape[0]], 1., crop=False)
                    (x, y, w, h) = imutil.rect_fit_points(landmarks[i], )
                    r_ = np.array([x, y, w, h]).astype(dtype=np.int).tolist()
                    predictions['rectangles'][-1] = r_

                    (x, y, w, h) = r_
                    face = np.zeros((h, w, 3), dtype=np.float)
                    y_ = 0
                    x_ = 0
                    if y + h > img.shape[0]:
                        h = img.shape[0] - y
                    elif y < 0:
                        y_ = -y
                        y = 0
                        h = h - y_
                    if x + w > img.shape[1]:
                        w = img.shape[1] - x
                    elif x < 0:
                        x_ = -x
                        x = 0
                        w = w - x_
                    face[y_:y_+h, x_:x_+w, :] = img[y:y+h, x:x+w, :]

                    face = cv2.resize(face, (48, 48), interpolation = cv2.INTER_NEAREST)
                    face = (face[:, :, 0:1] * 0.2126 + face[:, :, 1:2] * 0.7152 + face[:, :, 2:3] * 0.0722).reshape((48, 48))
                    #face_write = (face * 255.).astype(np.uint8)
                    #cv2.imwrite('./face'+str(i).zfill(3)+'.jpg', face_write)
                    facelist[i:i+1, :, :] = face
                    predictions['timing']['emoc_prepare'] += (time.time() - t_) * 1000

            if model=='alexnet-emoc':
                t_ = time.time()
                emoc_ = self.__get_emotion_classifier_create()
                emotions = emoc_.predict(facelist)
                predictions['emotions'] = (np.array(emotions)*1000).astype(dtype=np.int).tolist()
                predictions['timing']['emoc'] = (time.time() - t_) * 1000

        else:
            # Nothing is requested; useful for measuring overhead
            pass

        return predictions

if __name__== "__main__":
    face_app = FaceApplications()
    face_app.detect()

    r_ = Rectangle(12, 24, 48, 96)
    print(r_.eval())
    e_ = r_.to_extent()
    print(e_.eval())
    r_ = e_.to_rectangle()
    print(r_.eval())

