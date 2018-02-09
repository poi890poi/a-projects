import pydoc
import cv2
import math
import numpy as np
import scipy.io as sio
import scipy.stats
from scipy.misc import imresize
from skimage.feature import hog
from skimage import data, exposure

import argparse
import collections
import os.path
import sys
from uuid import uuid4

ARGS = None

RecursiveDirectoryWalker = collections.namedtuple('RecursiveDirectoryWalker', 'directory, iterator, pointer, history, filter')

class RecursiveDirectoryWalkerManager():
    def __init__(self):
        self.walkers = dict()

    def get_walker_create(self, directory):
        if directory not in self.walkers:
            walker = RecursiveDirectoryWalker(
                directory = directory,
                iterator = self.recursive_file_iterator(directory),
                pointer = 0,
                history = list(),
                filter = list(),
            )
            self.walkers[directory] = walker
        return self.walkers[directory]

    def recursive_file_iterator(self, directory):
        for entry in os.scandir(directory):
            if entry.is_dir():
                yield from self.recursive_file_iterator(entry.path)
            else:
                yield entry

    def get_a_file(self, directory, filters=None):
        walker = self.get_walker_create(directory)
        for f in walker.iterator:
            if filters:
                ext = os.path.splitext(f.name)[1].lower()
                if ext not in filters:
                    continue
            return f
        return None

ImageProcessorParameters = collections.namedtuple('ImageProcessorParameters', 'convert_gray')

class ImageProcessor():
    GRAY_NONE = 0
    GRAY_YCRCB = 1

    def __init__(self):
        self.minsize = 256
        self.maxsize = 256
        self.viewports = dict()

    def draw(self, img, wname):
        if wname not in self.viewports:
            cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
            self.viewports[wname] = 1
        size = np.array(img.shape)
        r = 1
        if size[0] < self.minsize or size[1] < self.minsize:
            r = max(self.minsize//size[0]+1, self.minsize//size[1]+1)
            size = size*r
        cv2.resizeWindow(wname, size[1], size[0])
        cv2.imshow(wname, img)

    def preprocess(self, img, imp=None):
        size = np.array(img.shape)
        r = 1.
        if size[0] > self.maxsize or size[1] > self.maxsize:
            r = min(self.maxsize/size[0], self.maxsize/size[1])
        size = ((size.astype('float32'))*r).astype('int16')
        img = imresize(img, size)
        
        if imp and imp.convert_gray==self.GRAY_YCRCB:
            # Convert to YCrCb and keep only Y channel.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            channels = cv2.split(img)
            img = channels[0]

        depth = 1
        if len(img.shape)==3:
            width, height, depth = img.shape
        else:
            width, height = img.shape
        
        # Denoise and equalize. Note 2018-02-09: Denoising benefits HAAR face detector significantly
        if depth==1:
            img = cv2.fastNlMeansDenoising(img)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img  = clahe.apply(img)
        elif depth==3:
            img = cv2.fastNlMeansDenoisingColored(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            channels = cv2.split(img)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img[:, :, 0] = clahe.apply(channels[0])
            img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        else:
            raise TypeError('ImageProcessor::preprocess() expects image of 1 or 3 channels')
        
        return img

    def gray2rgb(self, gray):
        # Clone single channel to RGB
        return np.repeat(gray[:, :, np.newaxis], 3, axis=2)

    def wait_user(self):
        # Key input must be handled in subclass
        raise NotImplementedError('ImageProcessor::wait_user() is not implmented')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for prefix in self.viewports:
            cv2.destroyWindow(prefix)


class DataGenerator(ImageProcessor):
    def __init__(self, args):
        self.args = args
        self.positive = None
        self.negative = None
        self.data = None
        self.label = None
        super(DataGenerator, self).__init__()
        self.dir_walk_mgr = RecursiveDirectoryWalkerManager()

    def gen(self, preview=False, imp=None):
        # Each call get a file from source directory.
        # Returns only when a valid sample is found (spawn() returns True),
        # or when there's no more sample in source directory
        while True:
            f = self.dir_walk_mgr.get_a_file(directory=self.args.source_dir, filters=['.jpg'])
            if f is None:
                print('No more file T_T')
                return

            img = cv2.imread(os.path.normpath(f.path), 1) # Load as RGB for compatibility
            if img is None:
                continue

            img = self.preprocess(img, imp)
            self.spawn(img, preview)

    def spawn(self, img, preview):
        raise NotImplementedError('DataGenerator::spawn() is not implmented')

    def preview(self, img, annotations, wname):
        # Annotations are context sensitive and must be implmented in subclass
        raise NotImplementedError('DataGenerator::preview() is not implmented')

    def negative_striding(self, img, positive, limit=4):
        # Stride through source image and crop to generate subset that does not contain positive sample
        pass


def prepare_dir(head, trail, create=True, empty=False):
    directory = norm_join_path(head, trail)
    if empty and os.path.isdir(directory):
        print()
        print('Removing directory:', directory)
        tmp = norm_join_path(head, hash_str(directory))
        os.rename(directory, tmp)
        os.removedirs(tmp)
    if create and not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def norm_join_path(head, trail):
    return os.path.normpath(os.path.join(head, trail))


class FaceGenerator(DataGenerator):
    def __init__(self, args):
        self.args = args
        super(FaceGenerator, self).__init__(args)
        self.positive_dir = prepare_dir(args.dest_dir, 'positive')
        self.negative_dir = prepare_dir(args.dest_dir, 'negative')
        
        self.face_cascade = None
        self.eye_cascade = None
        self.fseq = 0

    def spawn(self, img, preview):
        # Use opencv HAAR detector to get annotation for face
        # https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html

        t_size = [64, 96]

        if self.face_cascade is None: self.face_cascade = cv2.CascadeClassifier('./pretrained/haarcascades/haarcascade_frontalcatface.xml')
        if self.eye_cascade is None: self.eye_cascade = cv2.CascadeClassifier('./pretrained/haarcascades/haarcascade_eye.xml')
        faces = self.face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

        if len(faces):
            eyeslist = list()
            i = 0
            for (x, y, w, h) in faces:
                roi_gray = img[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                if len(eyes)==2:
                    eyeslist.append(eyes)
                    # Normalize face position using height of eye
                    ex1, ey1, ew1, eh1 = eyes[0]
                    ex2, ey2, ew2, eh2 = eyes[1]
                    # Normalize width and x position
                    eyedistance = abs(ex2 + ew2//2 - ex1 - ew1//2)
                    w = int(eyedistance * 2)
                    eyecenter = x + (ex1 + ew1//2 + ex2 + ew2//2)//2
                    x = int(eyecenter - w//2)
                    # Normalize height and y position
                    eyelevel = y + (ey1 + eh1//2 + ey2 + eh2//2)//2
                    h = int(w * t_size[1] / t_size[0])
                    y = int(eyelevel - h//3)
                    # TODO: Check if source ROI is too small

                    if not (w < t_size[0] or h < t_size[1] or y < 0 or y+h >= img.shape[1] or x < 0 or x+w >= img.shape[0]):
                        # Good sample. Save it
                        sys.stdout.write('@')
                        sys.stdout.flush()
                        faces[i] = [x, y, w, h]
                        sample = img[y:y+h, x:x+w, :]
                        sample = imresize(sample, np.flip(t_size, 0))
                        #entropy = scipy.stats.entropy(np.reshape(sample, (t_size[0]*t_size[1],)))
                        #print(sample.shape, entropy)
                        cv2.imwrite(os.path.join(self.positive_dir, str(self.fseq).zfill(4)+'.jpg'), sample)
                        self.fseq += 1
                        sample = np.flip(sample, 1)
                        cv2.imwrite(os.path.join(self.positive_dir, str(self.fseq).zfill(4)+'.jpg'), sample)
                        self.fseq += 1
                else:
                    # Not exactly 2 eyes. Confidence low
                    sys.stdout.write('x')
                    sys.stdout.flush()
                    faces[i] = [0, 0, 0, 0]
                i += 1
            if preview: self.preview(img, (faces, eyeslist), 'preview')
            return True

        sys.stdout.write('.')
        sys.stdout.flush()

        return False

    def preview(self, img, annotations, wname):
        img = self.gray2rgb(img)

        faces = annotations[0]
        eyeslist = annotations[1]
        i = 0
        for (x, y, w, h) in faces:
            if w and h:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 128, 128), 1)
                eyes = eyeslist[i]
                if len(eyes)==2:
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0,255,0), 1)
            i += 1

        self.draw(img, wname)
        self.wait_user() # Always wait for user input if preview is displayed

    def wait_user(self):
        while True:
            k = cv2.waitKeyEx(0)
            if k in (1, 27, -1): # esc
                print()
                print('User terminated :>')
                sys.exit()
            elif k in (13, 32): # enter, space
                self.gen(preview=True)
            elif k in (2490368, ): # up
                pass
            elif k in (2621440, ): # down
                pass
            elif k in (2424832,): # left
                pass
            elif k in (2555904,): # right
                pass
            else:
                print('unregistered key_code:', k)


def main():
    with FaceGenerator(ARGS) as dg:
        dg.gen(preview=ARGS.preview, imp=ImageProcessorParameters(
            convert_gray = ImageProcessor.GRAY_NONE,
        ))


if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Generate samples with supplied directory of source images""")
    parser.add_argument(
        '--source_dir',
        type=str,
        default='../../data/face/helen',
        help='Path to the source data.'
    )
    parser.add_argument(
        '--dest_dir',
        type=str,
        default='../../data/face/processed',
        help='Path to the processed data.'
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Display samples in window for preview.'
    )
    ARGS, unknown = parser.parse_known_args()
    if unknown:
        print(unknown)
        raise Exception('Unknown argument')
    main()