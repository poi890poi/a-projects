from shared.utilities import *
from shared.models import *
from shared.dataset import *
from face.dlibdetect import FaceDetector
from mtcnn import detect_face as mtcnn_detect
from emotion.emotion_recognition import EmotionRecognition
from facer.emotion import EmotionClassifier

import os.path
import pprint
import time
import json

import threading
import base64

import cv2
import numpy as np
import tensorflow as tf

shape_raw = (48, 48, 3)
shape_flat = (np.prod(shape_raw),)
shape_image = (12, 12, 3)
n_class = 2
data_size = 2000

def get_data():
    # Hyperparameters
    prep_denoise = False
    prep_equalize = False

    train_count = 0
    val_count = data_size
    total_count = train_count + val_count

    train_positive_count = train_count//2
    train_negative_count = train_count - train_positive_count
    val_positive_count = val_count//2
    val_negative_count = val_count - val_positive_count
    positive_count = train_positive_count + val_positive_count
    negative_count = train_negative_count + val_negative_count

    a = np.zeros(shape=(total_count,)+shape_raw, dtype=np.float32) # Data
    b = np.full(shape=[total_count, n_class], fill_value=-1, dtype=np.int) # Labels
    c = np.zeros(shape=[total_count, n_class], dtype=np.int) # Label assignment
    c[:] = [1, 0]
    c[0:train_positive_count] = [0, 1]
    c[train_count:train_count+val_positive_count] = [0, 1]

    train_data = a[0:train_count, :, :, :]
    train_labels = b[0:train_count, :]
    val_data = a[train_count:total_count, :, :, :]
    val_labels = b[train_count:total_count, :]

    # Load positive data from dataset
    count = positive_count
    face_index = 0
    while count:
        f = DirectoryWalker().get_a_file(directory='../data/face/val/positive', filters=['.jpg'])
        img = cv2.imread(f.path, 1)
        if img is None:
            continue
        src_shape = img.shape
        height, width, *rest = img.shape
        (x, y, w, h) = ImageUtilities.rect_fit_ar([0, 0, width, height], [0, 0, width, height], 1, mrate=1., crop=True)
        if w>0 and h>0:
            pass
        else:
            continue

        #face = img[y:y+h, x:x+w, :]
        face = ImageUtilities.transform_crop((x, y, w, h), img, r_intensity=0., p_intensity=0.)
        face = imresize(face, shape_raw[0:2])
        face = ImageUtilities.preprocess(face, convert_gray=None, equalize=prep_equalize, denoise=prep_denoise)
        #face = tf.image.per_image_standardization(face)
        face = np.array(face, dtype=np.float32)/255

        for i in range(len(b)):
            if np.array_equal(c[i], [0, 1]) and not np.array_equal(b[i], c[i]):
                break
        #print(i, 'positive', imgpath)
        a[i, :, :, :] = face
        b[i] = c[i]

        count -= 1

    # Load negative data
    count = negative_count
    while count:
        f = DirectoryWalker().get_a_file(directory='../data/face/val/negative', filters=['.jpg'])
        img = cv2.imread(f.path, 1)
        height, width, *rest = img.shape
        crop = (height - width)//2
        img = img[crop:crop+width, :, :]
        img = imresize(img, shape_raw[0:2])
        img = ImageUtilities.preprocess(img, convert_gray=None, equalize=prep_equalize, denoise=prep_denoise)
        #img = tf.image.per_image_standardization(img)
        img = np.array(img, dtype=np.float32)/255

        for i in range(len(b)):
            if np.array_equal(c[i], [1, 0]) and not np.array_equal(b[i], c[i]):
                break
        #print(i, 'negative', f.path)
        a[i, :, :, :] = img
        b[i] = c[i]

        count -= 1

    #print(train_data.shape)
    #print(train_labels.shape)
    #print(val_data.shape)
    #print(val_labels.shape)
    return (train_data, train_labels, val_data, val_labels)

# python fr-test.py --test=val --preview --model=../models/cascade/checkpoint/model.ckpt
# python fr-test.py --test=hnm --model=../models/cascade/checkpoint/model.ckpt --subset=train --count=2400

def val_stats(args):
    classifier = FaceClassifier()
    classifier.init(args)

    dir_pt = 0
    directories = [
        ['../data/face/val/positive', [0., 1.]],
        ['../data/face/val/negative', [1., 0.]],
    ]

    batch_size = 1200
    while True:
        val_data = list()
        val_labels = list()

        directory = directories[dir_pt][0]
        label = directories[dir_pt][1]

        while True:
            f = DirectoryWalker().get_a_file(directory=directory, filters=['.jpg'])
            if f is None: break
            img = cv2.imread(f.path, 1)
            if img is None:break

            height, width, *rest = img.shape
            (x, y, w, h) = ImageUtilities.rect_fit_ar([0, 0, width, height], [0, 0, width, height], 1, mrate=1., crop=True)
            face = ImageUtilities.transform_crop((x, y, w, h), img, r_intensity=0., p_intensity=0.)
            face = imresize(face, [48, 48])
            face = np.array(face, dtype=np.float)/255

            val_data.append(face)
            val_labels.append(label)
            if len(val_data)>=batch_size:
                break
        
        if len(val_data)==0:
            # No more data in the directory; proceed to next directory
            dir_pt += 1
            if dir_pt>=len(directories):
                break
        else:
            val_labels = np.array(val_labels)
            val_data = np.array(val_data)
            classifier.val(val_data, val_labels)

def val_preview(args):
    print()
    subsamples_per_image = 32
    preview = args.preview
    print(preview)
    
    path_source = '../data/face/demo'

    classifier = FaceClassifier()
    classifier.init(args.model)

    while True:
        f = DirectoryWalker().get_a_file(directory=path_source, filters=['.jpg'])

        inpath = f.path
        img = cv2.imread(inpath, 1)
        img = ImageUtilities.preprocess(img, convert_gray=None, equalize=True, denoise=False, maxsize=-1)

        target_class = 1
        rects, predictions, timing = classifier.multi_scale_detection(img)
        if len(predictions):
            scores = np.array(predictions)[:, target_class:target_class+1].reshape((-1,))
            nms = tf.image.non_max_suppression(np.array(rects), scores, max_output_size=99999)
            print('face detectd', len(nms.eval()))
            for index, value in enumerate(nms.eval()):
                rect = rects[value]

                face = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :]
                face = imresize(face, shape_raw[0:2])
                if preview: cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)

        if preview:
            canvas = ViewportManager().open('preview', shape=img.shape, blocks=(1, 2))
            ViewportManager().put('preview', img, (0, 0))
            ViewportManager().update('preview')
            
            k = ViewportManager().wait_key()
            if k in (ViewportManager.KEY_ENTER, ViewportManager.KEY_SPACE):
                pass

def val(args):
    fxpress = EmotionRecognition()
    fxpress.build_network()

    directory = '../data/face/fer2013/publictest'
    while True:
        f = DirectoryWalker().get_a_file(directory=directory, filters=['.jpg'])
        if f is None: break
        components = os.path.split(f.path)
        label = int(os.path.split(components[0])[1])

        img = cv2.imread(f.path, 0)
        if img is None:break

        emotion = np.argmax(fxpress.predict(img)[0])
        print('emotion', label, emotion)

def val_mtcnn(args):
    classifier = FaceClassifier()
    classifier.init(args)

    dir_pt = 0
    directories = [
        ['../data/face/val/positive', [0., 1.]],
        ['../data/face/val/negative', [1., 0.]],
    ]

    count_val = 0
    count_correct = 0
    count_true_positive = 0
    count_positive = 0

    while True:
        directory = directories[dir_pt][0]
        label = directories[dir_pt][1]
        while True:
            f = DirectoryWalker().get_a_file(directory=directory, filters=['.jpg'])
            if f is None: break
            img = cv2.imread(f.path, 1)
            if img is None:break

            # dlib detection
            height, width, *rest = img.shape
            (x, y, w, h) = ImageUtilities.rect_fit_ar([0, 0, width, height], [0, 0, width, height], 1, mrate=1., crop=True)
            face = ImageUtilities.transform_crop((x, y, w, h), img, r_intensity=0., p_intensity=0.)
            face = imresize(face, [48, 48])
            gray = ImageUtilities.preprocess(img, convert_gray=cv2.COLOR_RGB2YCrCb, equalize=False, denoise=False, maxsize=-1)

            dlib_rects, landmarks = FaceDetector().detect(gray)
            
            count_val += 1
            if label[1]>label[0]:
                count_positive += 1
            if len(dlib_rects) and label[1]>label[0]:
                count_true_positive += 1
                count_correct += 1
            if len(dlib_rects)==0 and label[0]>label[1]:
                count_correct += 1
            if count_val%1000==999:
                print('Precision:', count_correct/count_val)
                print('Recall:', count_true_positive/count_positive)
                
            # MTCNN detection
            """height, width, *rest = img.shape
            (x, y, w, h) = ImageUtilities.rect_fit_ar([0, 0, width, height], [0, 0, width, height], 1, mrate=1., crop=True)
            face = ImageUtilities.transform_crop((x, y, w, h), img, r_intensity=0., p_intensity=0.)
            face = imresize(face, [48, 48])

            minsize = 40 # minimum size of face
            threshold = [0.6, 0.7, 0.9]  # three steps's threshold
            factor = 0.709 # scale factor
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            bounding_boxes, points = mtcnn_detect.detect_face(face, minsize, classifier.pnet, classifier.rnet, classifier.onet, threshold, factor)
            
            count_val += 1
            if label[1]>label[0]:
                count_positive += 1
            if len(bounding_boxes) and label[1]>label[0]:
                count_true_positive += 1
                count_correct += 1
            if len(bounding_boxes)==0 and label[0]>label[1]:
                count_correct += 1
            if count_val%1000==999:
                print('Precision:', count_correct/count_val)
                print('Recall:', count_true_positive/count_positive)"""

        print('next directory')
        dir_pt += 1
        if dir_pt>=len(directories):
            break

        print('Precision:', count_correct/count_val)
        print('Recall:', count_true_positive/count_positive)

def val_old(args):
    if args.preview:
        val_preview(args)
    else:
        val_stats(args)

target_class = 1

def hnm(args):
    # Use COCO annotations to exclude people
    print()
    count = args.count
    if count==0: count = 2400
    subsamples_per_image = 32
    preview = False

    use_dlib = True
    
    subset = args.subset
    year = '2017'
    path_anno = '../data/coco/annotations_trainval2017/annotations/instances_'+subset+year+'.json'
    path_source = '../data/coco/'+subset+year+'/'+subset+year
    path_target = '../data/face/'+subset+'/negative/coco'

    
    print('Restoring model', args.model)
    classifier = FaceClassifier()
    classifier.init(args.model)
    print('done')
    print()

    print('Loading annotations...', path_anno)
    anno = None
    with open(path_anno, 'r') as f:
        anno = json.load(f)
    print()

    categories = dict()
    for ele in anno['categories']:
        categories[ele['id']] = ele

    def get_image_by_id(id):
        for ele in anno['images']:
            if ele['id']==id:
                return ele
        return None

    def get_anno_by_image_id(id):
        _annos = list()
        for ele in anno['annotations']:
            if ele['image_id']==id:
                _annos.append(ele)
        return _annos

    def rect_overlap(rect1, rect2):
        if rect2[0] >= rect1[0]+rect1[2] or rect2[1] >= rect1[1]+rect1[3] or rect1[0] >= rect2[0]+rect2[2] or rect1[1] >= rect2[1]+rect2[3]:
            return 0
        return 1

    print('annotated images', len(anno['images']))
    for ele in anno['images']:
        id = ele['id']
        file_name = ele['file_name']
        _annos = get_anno_by_image_id(id) 
        #print(_annos)

        inpath = path_source + '/' + file_name
        img = cv2.imread(inpath, 1)
        img = ImageUtilities.preprocess(img, convert_gray=None, equalize=False, denoise=False, maxsize=-1)

        rects_exclude = list()
        if use_dlib:
            # Use dlib face detector to exclude face
            gray = ImageUtilities.preprocess(img, convert_gray=cv2.COLOR_RGB2YCrCb, equalize=False, denoise=False, maxsize=-1)
            dlib_rects, landmarks = FaceDetector().detect(gray)
            for r_ in dlib_rects:
                bbox = np.array(ImageUtilities.rect_to_bb(r_), dtype=np.int).tolist()
                rects_exclude.append(bbox)
        else:
            # Use COCO annotations to exclude people
            for _object in _annos:
                category_id = _object['category_id']
                category = categories[category_id]
                #print(category)
                bbox = _object['bbox']
                bbox = np.array(bbox, dtype=np.int).tolist()
                if category['supercategory']=='person':
                    if preview: cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 2)
                    rects_exclude.append(bbox)

        rects, predictions, timing = classifier.multi_scale_detection(img)

        print('detected', len(predictions))

        if len(predictions)<=0:
            continue
        
        scores = np.array(predictions)[:, target_class:target_class+1].reshape((-1,))
        nms = tf.image.non_max_suppression(np.array(rects), scores, max_output_size=subsamples_per_image)
        print('nms', len(nms.eval()))
        for index, value in enumerate(nms.eval()):
            rect = rects[value]

            person_overlap = False
            for exclude in rects_exclude:
                if rect_overlap(rect, exclude):
                    person_overlap = True
                    break
            
            if not person_overlap:
                face = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :]
                face = imresize(face, shape_raw[0:2])
                outpath = path_target + '/' + ImageUtilities.hash(face) + '.jpg'
                cv2.imwrite(outpath, face)
                print('imwrite', outpath, count)
                count -= 1
                if count<=0: return
                if preview: cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)

        if preview:
            canvas = ViewportManager().open('preview', shape=img.shape, blocks=(1, 2))
            ViewportManager().put('preview', img, (0, 0))
            ViewportManager().update('preview')
            
            k = ViewportManager().wait_key()
            if k in (ViewportManager.KEY_ENTER, ViewportManager.KEY_SPACE):
                pass

def predict(args):
    path_source = '../data/coco/val2017/val2017'

    classifier = FaceClassifier()
    classifier.init(args.model)

    while True:
        f = DirectoryWalker().get_a_file(directory=path_source, filters=['.jpg'])
        img = cv2.imread(f.path, 1)
        if img is None:
            continue

        rects, predictions, timing = classifier.multi_scale_detection(img)

        for rect in rects:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)

        canvas = ViewportManager().open('preview', shape=img.shape, blocks=(1, 2))
        ViewportManager().put('preview', img, (0, 0))
        ViewportManager().update('preview')
        
        k = ViewportManager().wait_key()
        if k in (ViewportManager.KEY_ENTER, ViewportManager.KEY_SPACE):
            pass

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class FaceClassifier(metaclass=Singleton):
    def __init__(self):
        self.model = None
        self.sess_mtcnn = None
        self.fxpress = None
        self.emoc = None

    def init(self, args=None):
        if self.model is None:
            if args is None:
                self.model = FaceCascade({
                    'mode': 'INFERENCE',
                    'model_dir': '../models/cascade',
                    'ckpt_prefix': './server/models/12-net/model.ckpt',
                    'cascade': 12,
                })
            else:
                self.model = FaceCascade({
                    'mode': 'INFERENCE',
                    'model_dir': '../models/cascade',
                    'ckpt_prefix': args.model,
                    'cascade': args.cascade,
                })
        self.threshold = 0.99

        self.count_val = 0
        self.count_correct = 0
        self.count_positive = 0
        self.count_true_positive = 0

        if self.sess_mtcnn is None:
            self.sess_mtcnn = tf.Session()
            self.pnet, self.rnet, self.onet = mtcnn_detect.create_mtcnn(self.sess_mtcnn, None)

        """if self.fxpress is None:
            self.fxpress = EmotionRecognition()
            self.fxpress.build_network()"""

        if self.emoc is None:
            self.emoc = EmotionClassifier()
            self.emoc.build_network(args)

    def val(self, val_data, val_labels):
        # Detect with self-trained 12-net
        feed_dict = {self.model.x: val_data.reshape((-1,)+shape_flat)}
        predictions = self.model.sess.run(self.model.y, feed_dict)

        i = 0
        for p in predictions:
            ground_truth = val_labels[i]
            if np.argmax(ground_truth)==1:
                self.count_positive += 1
                if np.argmax(p)==1:
                    self.count_true_positive += 1

            if np.argmax(p)==np.argmax(ground_truth):
                self.count_correct += 1

            i += 1

        self.count_val += len(predictions)

        print()
        #print('feed', len(predictions))
        #print('total', self.count_val, 'positive', self.count_positive)
        print('Precision:', self.count_correct/self.count_val)
        print('Recall:', self.count_true_positive/self.count_positive)

    def multi_scale_detection(self, img, expanding_rate=1.2, initial=48, stride=12):
        source = np.copy(img)

        timing = dict()
        timing['preprocess'] = 0
        timing['detect'] = 0

        window_size = np.array([initial, initial], dtype=np.int)
        stride_x = np.array([0, stride], dtype=np.int)
        stride_y = np.array([stride, 0], dtype=np.int)

        contracting_rate = 1./expanding_rate
        min_size = window_size[0]*2

        pyramid = list()
        pyramid.append(img)
        pyramid_level = 0
        window_list = list()
        pos_list = list()

        time_start = time.time()
        
        height, width, *_ = img.shape
        print(height, width)
        height_, width_, *_ = img.shape

        while True:

            window_pos = np.array([0, 0], dtype=np.int)
            while True:
                while True:
                    #print(window_pos)
                    # Get projection from source image
                    if window_pos[1]+window_size[1] >= width_ or window_pos[0]+window_size[0] >= height_:
                        break

                    face = np.copy(pyramid[pyramid_level][window_pos[0]:window_pos[0]+window_size[0], window_pos[1]:window_pos[1]+window_size[1], :], shape_raw[0:2])
                    window_list.append(face)
                    w = window_pos
                    rect = np.concatenate((np.flip(w, axis=0), np.flip(window_size, axis=0)), axis=0)
                    rect = (rect*width/width_).astype(np.int).tolist()
                    pos_list.append(rect)

                    window_pos += stride_x
                    if window_pos[1]+window_size[1] >= width_:
                        window_pos[1] = 0
                        window_pos += stride_y
                        break
                if window_pos[0]+window_size[0] >= height_:
                    break

            height_ = int(height_*contracting_rate)
            width_ = int(width_*contracting_rate)
            pyramid.append(imresize(pyramid[pyramid_level], [height_, width_]))
            pyramid_level += 1
            if width_ < min_size:
                break

        print(len(window_list), 'subsamples')
        val_data = np.array(window_list, dtype=np.float32)/255
        reshaped = val_data.reshape((-1,)+shape_flat)
        time_diff = time.time() - time_start
        timing['crop'] = time_diff*1000

        time_start = time.time()
        feed_dict = {self.model.x: val_data.reshape((-1,)+shape_flat)}
        predictions = self.model.sess.run(self.model.y, feed_dict)
        time_diff = time.time() - time_start
        #print(time_diff*1000, 'ms')
        #print(time_diff*1000/len(window_list), 'ms per window')
        timing['cnn'] = time_diff*1000
        timing['window_count'] = len(window_list)

        rects_ = list()
        predictions_ = list()
        i = 0
        for p in predictions:
            if np.argmax(p)==1:
                rects_.append(pos_list[i])
                predictions_.append(p.tolist())
            i += 1

        return (rects_, predictions_, timing)
        
    def detect(self, media):
        timing = dict()
        result = dict({
            'mtcnn': list(),
            'mtcnn_5p': list(),
            'emotions': list(),
        })
        #print(media)
        src_img = None
        if 'content' in media:
            bindata = base64.b64decode(media['content'].encode())
            src_img = cv2.imdecode(np.frombuffer(bindata, np.uint8), 1)

        if src_img is not None:
            #print(img.shape)
            src_shape = src_img.shape
            
            time_start = time.time()
            gray = ImageUtilities.preprocess(src_img, convert_gray=cv2.COLOR_RGB2YCrCb, equalize=False, denoise=False, maxsize=384)
            time_diff = time.time() - time_start
            timing['preprocess'] = time_diff*1000
            #print('preprocess', time_diff)

            processed_shape = gray.shape
            mrate = [processed_shape[0]/src_shape[0], processed_shape[1]/src_shape[1]]

            time_start = time.time()
            rects, landmarks = FaceDetector().detect(gray)
            time_diff = time.time() - time_start
            timing['detect'] = time_diff*1000
            #print('hog+svm detect', time_diff)

            time_start = time.time()
            facelist = list()
            rects_ = list()
            predictions_ = list()
            # Crop faces from source image
            for rect in rects:
                face = None
                (x, y, w, h) = ImageUtilities.rect_to_bb(rect, mrate=mrate)
                height, width, *rest = src_img.shape
                (x, y, w, h) = ImageUtilities.rect_fit_ar([x, y, w, h], [0, 0, width, height], 1., mrate=1.)
                if w>0 and h>0:
                    face = ImageUtilities.transform_crop((x, y, w, h), src_img, r_intensity=0., p_intensity=0.)
                    face = imresize(face, shape_raw[0:2])
                    #face = ImageUtilities.preprocess(face, convert_gray=None)
                if face is not None:
                    facelist.append(face)
                    rects_.append([x, y, w, h])
                    predictions_.append([0., 0.])
            val_data = np.array(facelist, dtype=np.float32)/255
            reshaped = val_data.reshape((-1,)+shape_flat)
            time_diff = time.time() - time_start
            timing['crop'] = time_diff*1000
            #print('prepare data for cnn', time_diff)

            # MTCNN
            img = ImageUtilities.preprocess(src_img, convert_gray=None, equalize=False, denoise=False, maxsize=384)
            mrate_ = src_shape[0]/img.shape[0]
            time_start = time.time()
            minsize = 40 # minimum size of face
            threshold = [0.6, 0.7, 0.9]  # three steps's threshold
            factor = 0.709 # scale factor
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bounding_boxes, points = mtcnn_detect.detect_face(img, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
            
            if len(bounding_boxes):
                points = np.array(points) * mrate_
                points = points.reshape(2, -1)
                points = np.transpose(points)
                points = points.reshape((len(bounding_boxes), -1, 2))

            time_diff = time.time() - time_start
            timing['mtcnn'] = time_diff*1000
            timing['emotion'] = 0
            print()
            print()
            print('result len', len(bounding_boxes), len(points))
            nrof_faces = bounding_boxes.shape[0]
            for i, b in enumerate(bounding_boxes):
                r_ = (np.array([b[0], b[1], b[2]-b[0], b[3]-b[1]])*mrate_).astype(dtype=np.int).tolist()
                result['mtcnn'].append(r_ + [int(b[4]*1000),])
                result['mtcnn_5p'].append(points[i].astype(dtype=np.int).tolist())
                #rects_.append(r_)
                #predictions_.append([0., 2.])

                # Facial Expression
                time_start = time.time()
                (x, y, w, h) = ImageUtilities.rect_fit_ar(r_, [0, 0, src_shape[1], src_shape[0]], 1., crop=False)
                if w>0 and h>0:
                    face = ImageUtilities.transform_crop((x, y, w, h), src_img, r_intensity=0., p_intensity=0.)
                    #face = imresize(face, shape_raw[0:2])
                    #face = ImageUtilities.preprocess(face, convert_gray=cv2.COLOR_RGB2YCrCb, equalize=False, denoise=False)
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite('./face.jpg', face)
                    #face = np.array(face, dtype=np.float32)/255
                    face = cv2.resize(face, (48, 48), interpolation = cv2.INTER_CUBIC) / 255.
                if face is not None:
                    #emotions = self.fxpress.predict(face)[0]
                    emotions = self.emoc.predict(face)
                    print('emotion', face.shape, emotions)
                    result['emotions'].append((np.array(emotions)*1000).astype(dtype=np.int).tolist())
                time_diff = time.time() - time_start
                timing['emotion'] += time_diff*1000
                    
                print('rect', b, r_)
            print('emotions', result['emotions'])
            print('mtcnn_5p', result['mtcnn_5p'])
            print()
            print()

            # Self-trained cascade face detection
            img = ImageUtilities.preprocess(src_img, convert_gray=None, equalize=False, denoise=False, maxsize=384)
            ms_rects, ms_predictions, ms_timing = self.multi_scale_detection(img, expanding_rate=1.2, stride=12)
            mrate_ = src_shape[0]/img.shape[0]
            timing['cnn'] = ms_timing['cnn']
            timing['window_count'] = ms_timing['window_count']

            use_nms = True
            if len(ms_predictions):
                if use_nms:
                    # Apply non-maximum-supression
                    scores = np.array(ms_predictions)[:, target_class:target_class+1].reshape((-1,))
                    nms = tf.image.non_max_suppression(np.array(ms_rects), scores, iou_threshold=0.5, max_output_size=99999)
                    for index, value in enumerate(nms.eval()):
                        r_ = (np.array(ms_rects[value])*mrate_).astype(dtype=np.int).tolist()
                        p_ = ms_predictions[value]
                        rects_.append(r_)
                        predictions_.append(p_)
                else:
                    for index, p_ in enumerate(ms_predictions):
                        r_ = (np.array(ms_rects[index])*mrate_).astype(dtype=np.int).tolist()
                        rects_.append(r_)
                        predictions_.append(p_)

            """time_start = time.time()
            feed_dict = {self.model.x: val_data.reshape((-1,)+shape_flat)}
            predictions = self.model.sess.run(self.model.y, feed_dict)
            time_diff = time.time() - time_start
            timing['cnn'] = time_diff*1000
            #print('cnn classify', time_diff, len(facelist))
            #print('predictions', predictions)"""

            return (rects_, predictions_, timing, result)
            