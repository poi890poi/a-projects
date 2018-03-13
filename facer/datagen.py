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

def gen(args):
    target_dir = '../data/face/val/positive/wiki'

    source_name = 'wiki'

    components = os.path.split(source_list[source_name])
    topdir = components[0]

    dset = WikiImdb()
    dset.load_annotations(source_list[source_name], source_name)

    i = 0
    while True:
        anno = dset.get_face(i)
        for imgpath in anno[3]:
            imgpath = os.path.normpath(os.path.join(topdir, imgpath))
            img = cv2.imread(imgpath, 1)
            if img is None:
                continue
            height, width, *rest = img.shape
            if height<96 or width<64:
                # Source image too small
                continue
            
            src_shape = img.shape
            gray = ImageUtilities.preprocess(img, convert_gray=cv2.COLOR_RGB2YCrCb, equalize=False, denoise=False, maxsize=800)
            processed_shape = gray.shape
            mrate = [processed_shape[0]/src_shape[0], processed_shape[1]/src_shape[1]]
            print('resizing rate', mrate)

            rects, landmarks = FaceDetector().detect(gray)

            for rect in rects:
                (x, y, w, h) = ImageUtilities.rect_to_bb(rect, mrate=mrate)
                height, width, *rest = img.shape
                (x, y, w, h) = ImageUtilities.rect_fit_ar([x, y, w, h], [0, 0, width, height], 2/3, mrate=1.)
                if w>0 and h>0:
                    face = ImageUtilities.transform_crop((x, y, w, h), img, r_intensity=0., p_intensity=0.)
                    face = imresize(face, [96, 64])
                    face = ImageUtilities.preprocess(face, convert_gray=None)

                    filename = target_dir + '/' + ImageUtilities.hash(face) + '.jpg'
                    cv2.imwrite(filename, face)
                    print('saved', rect)
                else:
                    pass

                if args.preview: cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            dob = WikiImdb.from_ml_date(anno[1])
            age = WikiImdb.age(dob, anno[2])
            print(anno[1], dob, anno[2], age/365, anno[4])
            for roi in anno[6]:
                roi = roi.astype(np.int)
                if args.preview: cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)

            if args.preview:
                canvas = ViewportManager().open('preview', shape=img.shape, blocks=(1, 2))
                ViewportManager().put('preview', img, (0, 0))
                ViewportManager().put('preview', face, (0, 1))
                ViewportManager().update('preview')
                
                k = ViewportManager().wait_key()
                if k in (ViewportManager.KEY_ENTER, ViewportManager.KEY_SPACE):
                    pass
        i += 1

def gen_sof(args):
    target_dir = '../data/face/val/positive/set002'

    # Download data
    #DatasetDownloader().add_file('../data/face/sof', 'https://drive.google.com/file/d/0BwO0RMrZJCioaW5TdVJtOEtfYUk/view?usp=sharing', '.')

    source_name = 'sof'

    components = os.path.split(source_list[source_name])
    topdir = components[0]

    dset = SoF()
    dset.load_annotations(source_list[source_name], source_name)

    i = 0
    while True:
        print()
        anno = dset.get_face(i)
        if anno is None:
            break
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

        """
        rects, landmarks = FaceDetector().detect(gray)
        print('detected', rects, landmarks)
        #print(anno)
        for rect in rects:
            (x, y, w, h) = ImageUtilities.rect_to_bb(rect, mrate=mrate)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)"""

        print('annotations', anno['gender'], anno['age'], anno['glasses'], anno['scarf'], anno['rect'])
        (x, y, w, h) = anno['rect'].astype(dtype=np.int)
        if args.preview: cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        height, width, *rest = img.shape
        (x, y, w, h) = ImageUtilities.rect_fit_ar(anno['rect'].astype(dtype=np.int), [0, 0, width, height], 2/3, mrate=1.)
        if w>0 and h>0:
            if args.preview: cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            print('expanded roi out of bound')
            continue

        """height, width, *rest = img.shape
        (x, y, w, h) = ImageUtilities.rect_fit_ar(anno['rect'].astype(dtype=np.int), [0, 0, width, height], 2/3, mrate=1.25)
        if w>0 and h>0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            print('expanded roi out of bound')
            continue"""

        #face = img[y:y+h, x:x+w, :]
        face = ImageUtilities.transform_crop((x, y, w, h), img, r_intensity=0., p_intensity=0.)
        face = imresize(face, [96, 64])
        face = ImageUtilities.preprocess(face, convert_gray=None)

        filename = target_dir + '/' + ImageUtilities.hash(face) + '.jpg'
        cv2.imwrite(filename, face)

        #'(\w{4})_(\d{5})_([mfMF])_(\d{2})(_([ioIO])_(fr|nf)_(cr|nc)_(no|hp|sd|sr)_(\d{4})_(\d)_(e0|en|em)_(nl|Gn|Gs|Ps)_([oemh]))*\.jpg'
        if args.preview:
            canvas = ViewportManager().open('preview', shape=img.shape, blocks=(1, 2))
            ViewportManager().put('preview', img, (0, 0))
            ViewportManager().put('preview', face, (0, 1))
            ViewportManager().update('preview')

            k = ViewportManager().wait_key()
            if k in (ViewportManager.KEY_ENTER, ViewportManager.KEY_SPACE):
                pass

        i += 1

    return