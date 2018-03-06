from shared.utilities import *
from shared.dataset import *
from face.dlibdetect import FaceDetector

import os.path

import cv2
import numpy as np

source_list = {
    'wiki': '../data/face/wiki-face/extracted/wiki/wiki.mat',
    'imdb': '../data/face/imdb-crop/extracted/imdb_crop/imdb.mat',
    'sof': '../data/face/sof/images/metadata.mat',

}

def run(args):
    # Download data
    #DatasetDownloader().add_file('../data/face/sof', 'https://drive.google.com/file/d/0BwO0RMrZJCioaW5TdVJtOEtfYUk/view?usp=sharing', '.')

    source_name = 'sof'

    components = os.path.split(source_list[source_name])
    topdir = components[0]

    dset = SoF()
    dset.load_annotations(source_list[source_name], source_name)

    for i in range(64):
        print()
        anno = dset.get_face(i*64)
        #print(anno)
        filename = '_'.join((anno['file_prefix'], 'e0_nl_o'))
        filename = '.'.join((filename, 'jpg'))
        imgpath = os.path.normpath(os.path.join(topdir, filename))
        print(imgpath)

        img = cv2.imread(imgpath, 1)
        src_shape = img.shape
        gray = ImageUtilities.preprocess(img, convert_gray=cv2.COLOR_RGB2YCrCb, maxsize=800)
        processed_shape = gray.shape
        mrate = [processed_shape[0]/src_shape[0], processed_shape[1]/src_shape[1]]

        rects, landmarks = FaceDetector().detect(gray)
        print('detected', rects, landmarks)
        #print(anno)

        for rect in rects:
            (x, y, w, h) = ImageUtilities.rect_to_bb(rect, mrate=mrate)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        print(anno['rect'])
        (x, y, w, h) = anno['rect'].astype(dtype=np.int)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #'(\w{4})_(\d{5})_([mfMF])_(\d{2})(_([ioIO])_(fr|nf)_(cr|nc)_(no|hp|sd|sr)_(\d{4})_(\d)_(e0|en|em)_(nl|Gn|Gs|Ps)_([oemh]))*\.jpg'
        canvas = ViewportManager().open('preview', shape=img.shape, blocks=(1, 2))
        ViewportManager().put('preview', img, (0, 0))
        ViewportManager().put('preview', gray, (0, 1))
        ViewportManager().update('preview')

        k = ViewportManager().wait_key()
        if k in (ViewportManager.KEY_ENTER, ViewportManager.KEY_SPACE):
            pass

    return

    source_name = 'wiki'

    components = os.path.split(source_list[source_name])
    topdir = components[0]

    dset = WikiImdb()
    dset.load_annotations(source_list[source_name], source_name)

    for i in range(1024):
        anno = dset.get_face(i)
        for imgpath in anno[3]:
            imgpath = os.path.normpath(os.path.join(topdir, imgpath))
            img = cv2.imread(imgpath, 1)
            src_shape = img.shape
            gray = ImageUtilities.preprocess(img, convert_gray=cv2.COLOR_RGB2YCrCb, maxsize=800)
            processed_shape = gray.shape
            mrate = [processed_shape[0]/src_shape[0], processed_shape[1]/src_shape[1]]
            print('resizing rate', mrate)

            rects, landmarks = FaceDetector().detect(gray)
            print('detected', rects, landmarks)

            for rect in rects:
                (x, y, w, h) = ImageUtilities.rect_to_bb(rect, mrate=mrate)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            dob = WikiImdb.from_ml_date(anno[1])
            age = WikiImdb.age(dob, anno[2])
            print(anno[1], dob, anno[2], age/365, anno[4])
            for roi in anno[6]:
                roi = roi.astype(np.int)
                cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)

            canvas = ViewportManager().open('preview', shape=img.shape, blocks=(1, 2))
            ViewportManager().put('preview', img, (0, 0))
            ViewportManager().put('preview', gray, (0, 1))
            ViewportManager().update('preview')
            
            k = ViewportManager().wait_key()
            if k in (ViewportManager.KEY_ENTER, ViewportManager.KEY_SPACE):
                pass

