from mtcnn import detect_face as FaceDetector
from facer.emotion import EmotionClassifier
from shared.utilities import ImageUtilities as imutil
from shared.utilities import DirectoryWalker as dirwalker

import sys
import time
import numpy as np
import cv2
import tensorflow as tf
import os.path
import pathlib
import math

from queue import Queue, Empty
import threading

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

class DetectionTask():
    """
    A task is a container to hold input parameters and output results for detection.
    It's pushed into the synchronized queue to be staged for detection.
    A task only holds variables and does not do intensive computing.
    """
    def __init__(self, img, params):
        self.img = img
        self.params = params

class VisionCore(threading.Thread):
    """
    A core is a thread class that manages model and does actual detection.
    Multiple instances of a core class allocate multiple copies of Tensorflow session, if it's used.
    Cores are managed by 'applications' class objects.
    """
    def __init__(self, in_queue, out_queue):
        threading.Thread.__init__(self)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self._init_derived()

    def _init_derived(self):
        raise NotImplementedError('VisionCore::__init_derived() must be implemented in derived class')

    def run(self):
        raise NotImplementedError('VisionCore::run() must be implemented in derived class')

class FaceCore(VisionCore):
    """ The singleton that manages all detection """
    def _init_derived(self):
        self.__mtcnn = None
        self.__emoc = None

    def __get_detector_create(self):
        if self.__mtcnn is None:
            print()
            sys.stdout.write('Loading MTCNN model for face detection...')
            self.__mtcnn = {}
            self.__mtcnn['session'] = tf.Session()
            #self.__pnet, self.__rnet, self.__onet = FaceDetector.create_mtcnn(self.__mtcnn, None)
            self.__mtcnn['pnet'], self.__mtcnn['rnet'], self.__mtcnn['onet'] = FaceDetector.create_mtcnn(self.__mtcnn['session'], None)
            sys.stdout.write('done')
            print()
        return self.__mtcnn

    def __get_emotion_classifier_create(self):
        if self.__emoc is None:
            print()
            sys.stdout.write('Loading 4-layers AlexNet model for emotion classifier...')
            self.__emoc = EmotionClassifier()
            self.__emoc.build_network(None)
            sys.stdout.write('done')
            print()
        return self.__emoc

    def run(self):
        # Initialize required models and sessions
        if self.__mtcnn is None:
            self.__get_detector_create()
        if self.__emoc is None:
            self.__get_emotion_classifier_create()

        while True:
            #grabs data from queue
            task = self.in_queue.get() # queue::get() is blocking by default

            print(task.params)
            task.params['output_holder'].put({'predictions': self.detect(task.img, task.params)})

            #signals to queue job is done
            self.in_queue.task_done()

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

            # Prepare parameters
            res_cap = 448
            factor = 0.6
            interp = cv2.INTER_NEAREST
            if 'options' in service:
                options = service['options']
                if 'res_cap' in options:
                    res_cap = int(options['res_cap'])
                if 'factor' in options:
                    factor = float(options['factor'])
                if 'interp' in options:
                    if options['interp']=='LINEAR':
                        interp = cv2.INTER_LINEAR
                    elif options['interp']=='AREA':
                        interp = cv2.INTER_AREA
            if factor > 0.9: factor = 0.9
            elif factor < 0.45: factor = 0.45
            # For performance reason, resolution is hard-capped at 800, which is suitable for most applications
            if res_cap > 800: res_cap = 800
            elif res_cap <= 0: res_cap = 448 # In case client fail to initialize res_cap yet included options in request
            print('options', factor, interp)
            
            # This is a safe guard to avoid very large image,
            # for best performance, client is responsible to scale the image before upload
            resized, scale_factor = imutil.fit_resize(img, maxsize=(res_cap, res_cap), interpolation=interp)
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
            extents, landmarks = FaceDetector.detect_face(resized, 40, pnet, rnet, onet, threshold=[0.6, 0.7, 0.9], factor=factor, interpolation=interp)
            predictions['timing']['mtcnn'] = (time.time() - t_) * 1000

            # For testing, detect second time with ROIs (like in tracking mode)
            if len(extents):
                t_ = time.time()
                height, width, *_ = resized.shape
                #print()
                #print('shape', resized.shape)
                #print('extents', extents)
                # Expand extents
                for i, e in enumerate(extents):
                    w = e[2] - e[0]
                    h = e[3] - e[1]
                    extents[i] += np.array((-w, -h, w, h, 0), dtype=np.float)
                    e = extents[i]
                    extents[i] = np.array((max(e[0], 0), max(e[1], 0), min(e[2], width-1), min(e[3], height-1), e[4]), dtype=np.float)
                #print('expand', extents)
                # Group overlapped extents
                for iteration in range(16):
                    no_overlap = True
                    for i1, e1 in enumerate(extents):
                        if e1[4]==0: continue
                        for i2, e2 in enumerate(extents):
                            if e2[4]==0: continue
                            if i1!=i2:
                                if e2[0]>e1[2] or e2[1]>e1[3] or e2[2]<e1[0] or e2[3]<e1[1]:
                                    pass
                                else:
                                    no_overlap = False
                                    extents[i1] = np.array((min((e1[0], e2[0])), min((e1[1], e2[1])), max((e1[2], e2[2])), max((e1[3], e2[3])), 1), dtype=np.float)
                                    extents[i2] = np.array((0, 0, 0, 0, 0), dtype=np.float)
                    if no_overlap: break
                #print(type(extents[0]))
                #print('group', extents)
                predictions['timing']['prepare_roi'] = (time.time() - t_) * 1000
                t_ = time.time()
                for i, e in enumerate(extents):
                    if e[4] > 0:
                        e = e.astype(dtype=np.int)
                        roi = resized[e[1]:e[3], e[0]:e[2], :]
                        _extents, _landmarks = FaceDetector.detect_face(roi, 40, pnet, rnet, onet, threshold=[0.6, 0.7, 0.9], factor=factor, interpolation=interp)
                        #print()
                        #print(i, roi.shape)
                        #print(_extents)
                predictions['timing']['mtcnn_roi'] = (time.time() - t_) * 1000
                predictions['timing']['mtcnn_roi_total'] = predictions['timing']['prepare_roi'] + predictions['timing']['mtcnn_roi']

            if len(landmarks):
                landmarks = np.array(landmarks) * scale_factor
            """if len(extents):
                _ = np.array(landmarks) * scale_factor
                _ = _.reshape(2, -1)
                _ = np.transpose(_)
                _ = _.reshape((-1, len(extents), 2))
                landmarks = np.swapaxes(_, 0, 1)"""

            if model=='a-emoc':
                facelist = np.zeros((len(extents), 48, 48), dtype=np.float)
                predictions['timing']['emoc_prepare'] = 0
            
            print()
            print(extents)
            for i, e in enumerate(extents):
                #e_ = (Rectangle(r_[0], r_[1], r_[2], r_[3]).to_extent().eval() * scale_factor).astype(dtype=np.int).tolist()
                r_ = (np.array([e[0], e[1], e[2]-e[0], e[3]-e[1]])*scale_factor).astype(dtype=np.int).tolist()
                print('extents', i, e, r_)
                predictions['rectangles'].append(r_)
                predictions['confidences'].append(int(e[4]*1000))
                plist_ = landmarks[i].astype(dtype=np.int).tolist()
                predictions['landmarks'].append(plist_)
                if model=='a-emoc':
                    t_ = time.time()
                    #(x, y, w, h) = imutil.rect_fit_ar(r_, [0, 0, img.shape[1], img.shape[0]], 1., crop=False)
                    (x, y, w, h) = imutil.rect_fit_points(landmarks[i], )
                    r_ = np.array([x, y, w, h]).astype(dtype=np.int).tolist()
                    
                    # Return landmarks-corrected reactangles instead of MTCNN rectangle.
                    # The rectangles are used to subsample faces for emotion recognition.
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

                    face = cv2.resize(face, (48, 48), interpolation = interp)
                    face = (face[:, :, 0:1] * 0.2126 + face[:, :, 1:2] * 0.7152 + face[:, :, 2:3] * 0.0722).reshape((48, 48))
                    #face_write = (face * 255.).astype(np.uint8)
                    #cv2.imwrite('./face'+str(i).zfill(3)+'.jpg', face_write)
                    facelist[i:i+1, :, :] = face
                    predictions['timing']['emoc_prepare'] += (time.time() - t_) * 1000

            if model=='a-emoc':
                t_ = time.time()
                emoc_ = self.__get_emotion_classifier_create()
                emotions = emoc_.predict(facelist)
                predictions['emotions'] = (np.array(emotions)*1000).astype(dtype=np.int).tolist()
                predictions['timing']['emoc'] = (time.time() - t_) * 1000

        else:
            # Nothing is requested; useful for measuring overhead
            pass

        return predictions

    def align_dataset(self, directory=None):
        if directory is None:
            directory = '../data/face/fer2013_raw'

        stats = {
            'count': 0,
            'left': [0., 0.],
            'right': [0., 0.],
            'low': [0., 0.],
        }

        while True:
            f = dirwalker().get_a_file(directory=directory, filters=['.jpg'])
            if f is None: break

            img = cv2.imread(f.path, 1)
            if img is None:break
            img = (img.astype(dtype=np.float32))/255

            detector = self.__get_detector_create()
            pnet = detector['pnet']
            rnet = detector['rnet']
            onet = detector['onet']
            extents, landmarks = FaceDetector.detect_face(img, 48, pnet, rnet, onet, threshold=[0.6, 0.7, 0.9], factor=0.5)

            if len(landmarks) and len(landmarks[0]):
                warpped = self.align_face(img, landmarks[0])

                outpath = f.path.replace('fer2013_raw', 'fer2013_aligned')
                if outpath != f.path:
                    print(outpath)
                    outdir = os.path.split(outpath)[0]
                    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(outpath, warpped*255)

                """# Statitics of positions of eyes and nose
                stats['count'] += 1
                count = stats['count']

                stats['left'][0] = stats['left'][0]*(count-1)/count + left_eye[0]/count
                stats['left'][1] = stats['left'][1]*(count-1)/count + left_eye[1]/count
                stats['right'][0] = stats['right'][0]*(count-1)/count + right_eye[0]/count
                stats['right'][1] = stats['right'][1]*(count-1)/count + right_eye[1]/count
                stats['low'][0] = stats['low'][0]*(count-1)/count + low_center[0]/count
                stats['low'][1] = stats['low'][1]*(count-1)/count + low_center[1]/count

                print(stats)"""

            else:
                pass

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class VisionMainThread(metaclass=Singleton):
    """
    The singleton that manages all computer vision cores of a specific function block, e.g., face application, object detection, scene segmentation....
    Each function block (a set of related applications) maintains its own thread with status, input queue, output queue...
    Each function block (a set of related applications) spawns multiple threads as 'core' that do the actual detection.
    """
    def __init__(self):
        pass

class FaceApplications(VisionMainThread):
    """ This is for backward compatibility and the interface MUST NOT be changed. """
    def __init__(self):
        self.in_queue = Queue()
        self.out_queue = Queue()
        for i in range(2):
            t = FaceCore(self.in_queue, self.out_queue)
            t.setDaemon(True)
            t.start()
        print('FaceApplications singleton initialized')

    def detect(self, img, params):
        """ Queue the task in VisionMainThread """
        print('detect', params)
        t = DetectionTask(img, params)
        self.in_queue.put(t)

    @staticmethod
    def align_face(img, landmarks, intensity=0.5, sz=48, ortho=False, expand=1.0):
        # Normalize face pose and position with landmarks of 5 points: eyes, nose, mouth corners

        # Convert landmarks to numpy array
        left_eye = np.array(landmarks[0], dtype=np.float)
        right_eye = np.array(landmarks[1], dtype=np.float)
        nose = np.array(landmarks[2], dtype=np.float)
        mouth = (np.array(landmarks[3], dtype=np.float) + np.array(landmarks[4], dtype=np.float))/2.
        low_center = (np.array(landmarks[2], dtype=np.float) + np.array(landmarks[3], dtype=np.float) + np.array(landmarks[4], dtype=np.float)) / 3.

        # Calculate correction vectors for vertical and horizontal axes
        eye_center = (right_eye+left_eye) / 2.
        vv = nose - eye_center
        vh = (right_eye - left_eye)

        if ortho:
            vv = np.roll(vh, 1)
            vv *= 17. / 24. * expand
            vv[0] *= -1
            vh *= expand

        # Change length of correction vectors to be proportionate to face dimension
        vv_high = vv * 35. / 17.
        vv_low = vv * 13. / 17.

        # Calculate 4 corners for perspective transformation
        corners = np.array(
            [low_center - vv_high - vh,
            low_center - vv_high + vh,
            low_center + vv_low + vh,
            low_center + vv_low - vh,],
            dtype= np.float)

        """debug = np.array(
            [low_center - vv_high - vh,
            low_center - vv_high + vh,
            low_center + vv_low + vh,
            low_center + vv_low - vh,
            low_center,
            low_center + vh,
            low_center + vv,
            ],
            dtype= np.float)
        return debug"""

        # Perspective transformation
        rect = np.array(corners, dtype = "float32")
        dst = np.array([[0, 0], [sz-1, 0], [sz-1, sz-1], [0, sz-1]], dtype = "float32")
        rect = rect*intensity + dst*(1.-intensity)
        M = cv2.getPerspectiveTransform(rect, dst)

        return cv2.warpPerspective(img, M, (sz, sz), borderMode=cv2.BORDER_CONSTANT)

    def align_dataset(self, directory=None):
        pass

if __name__== "__main__":
    face_app = FaceApplications()
    face_app.detect()

    r_ = Rectangle(12, 24, 48, 96)
    print(r_.eval())
    e_ = r_.to_extent()
    print(e_.eval())
    r_ = e_.to_rectangle()
    print(r_.eval())

