from shared.utilities import *
from shared.models import *
from shared.dataset import *
from face.dlibdetect import FaceDetector

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
    classifier.init(args.model)

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
    if args.preview:
        val_preview(args)
    else:
        val_stats(args)

def hnm(args):
    print()
    count = args.count
    subsamples_per_image = 32
    preview = False
    
    subset = args.subset
    year = '2017'
    path_anno = '../data/coco/annotations_trainval2017/annotations/instances_'+subset+year+'.json'
    path_source = '../data/coco/'+subset+year+'/'+subset+year
    path_target = '../data/face/'+subset+'/negative/coco'

    print('Loading annotations...', path_anno)
    anno = None
    with open(path_anno, 'r') as f:
        anno = json.load(f)
    print('done')
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

    classifier = FaceClassifier()
    classifier.init(args.model)

    for ele in anno['images']:
        id = ele['id']
        file_name = ele['file_name']
        _annos = get_anno_by_image_id(id) 
        #print(_annos)

        inpath = path_source + '/' + file_name
        img = cv2.imread(inpath, 1)
        img = ImageUtilities.preprocess(img, convert_gray=None, equalize=True, denoise=False, maxsize=-1)

        rects_exclude = list()
        for _object in _annos:
            category_id = _object['category_id']
            category = categories[category_id]
            #print(category)
            bbox = _object['bbox']
            bbox = np.array(bbox, dtype=np.int).tolist()
            if category['supercategory']=='person':
                if preview: cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 2)
                rects_exclude.append(bbox)

        target_class = 1
        rects, predictions, timing = classifier.multi_scale_detection(img)

        if len(predictions)<=0:
            continue
        
        scores = np.array(predictions)[:, target_class:target_class+1].reshape((-1,))
        nms = tf.image.non_max_suppression(np.array(rects), scores, max_output_size=subsamples_per_image)
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

    def init(self, model_dir):
        if self.model is None:
            self.model = FaceCascade({
                'mode': 'INFERENCE',
                'model_dir': '../models/cascade',
                'ckpt_prefix': model_dir
            })
        self.threshold = 0.99

        self.count_val = 0
        self.count_correct = 0
        self.count_positive = 0
        self.count_true_positive = 0

    def val(self, val_data, val_labels):
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

    def multi_scale_detection(self, img, expanding_rate=1.3):
        source = np.copy(img)

        timing = dict()
        timing['preprocess'] = 0
        timing['detect'] = 0

        window_size = np.array([48, 48], dtype=np.int)
        window_stride = 12
        stride_x = np.array([0, window_stride], dtype=np.int)
        stride_y = np.array([window_stride, 0], dtype=np.int)
        
        contracting_rate = 1./expanding_rate
        min_size = window_size[0]*3

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
            if width_<min_size:
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
        print(time_diff*1000, 'ms')
        print(time_diff*1000/len(window_list), 'ms per window')
        timing['cnn'] = time_diff*1000

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
        #print(media)
        img = None
        if 'content' in media:
            bindata = base64.b64decode(media['content'].encode())
            img = cv2.imdecode(np.frombuffer(bindata, np.uint8), 1)

        if img is not None:
            return self.multi_scale_detection(img)

            #print(img.shape)
            src_shape = img.shape
            
            time_start = time.time()
            gray = ImageUtilities.preprocess(img, convert_gray=cv2.COLOR_RGB2YCrCb, equalize=False, denoise=False, maxsize=384)
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
            for rect in rects:
                face = None
                (x, y, w, h) = ImageUtilities.rect_to_bb(rect, mrate=mrate)
                height, width, *rest = img.shape
                (x, y, w, h) = ImageUtilities.rect_fit_ar([x, y, w, h], [0, 0, width, height], 1., mrate=1.)
                if w>0 and h>0:
                    face = ImageUtilities.transform_crop((x, y, w, h), img, r_intensity=0., p_intensity=0.)
                    face = imresize(face, shape_raw[0:2])
                    #face = ImageUtilities.preprocess(face, convert_gray=None)
                if face is not None:
                    facelist.append(face)
                    rects_.append([x, y, w, h])
            val_data = np.array(facelist, dtype=np.float32)/255
            reshaped = val_data.reshape((-1,)+shape_flat)
            time_diff = time.time() - time_start
            timing['crop'] = time_diff*1000
            #print('prepare data for cnn', time_diff)

            time_start = time.time()
            feed_dict = {self.model.x: val_data.reshape((-1,)+shape_flat)}
            predictions = self.model.sess.run(self.model.y, feed_dict)
            time_diff = time.time() - time_start
            timing['cnn'] = time_diff*1000
            #print('cnn classify', time_diff, len(facelist))
            #print('predictions', predictions)

            predictions_ = list()
            for p in predictions:
                predictions_.append(p.tolist())

            return (rects_, predictions_, timing)
            