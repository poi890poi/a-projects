import cv2
import numpy as np

from pathlib import Path
import argparse
import csv
import re
import os
from uuid import uuid4
import random
import pickle
import shutil
import tempfile
import hashlib
import wget, zipfile

import logging
from logging.handlers import RotatingFileHandler
import pprint

logfile = os.path.normpath(os.path.join(tempfile.gettempdir(), 'imgxform.log'))
logging.basicConfig(filename=logfile, level=logging.WARNING)
log = logging.getLogger()
handler = RotatingFileHandler(logfile, maxBytes=512*1024, backupCount=1)
log.addHandler(handler)

ARGS = None

def get_mutations(imgpath, count, intensity=1.0):
    # Get [count] mutations from image at [imgpath]
    proto = cv2.imread(imgpath, 0)
    width, height = proto.shape
    
    mutations = []
    for i in range(count):
        theta = 15. * intensity # Default rotation +/- 15 degrees as described in paper by Yann LeCun
        M = cv2.getRotationMatrix2D((width/2, height/2),
            random.uniform(-theta, theta), 1)

        # Rotate
        img = cv2.warpAffine(proto, M, (width, height), borderMode=cv2.BORDER_REPLICATE)

        # Perpective transformation
        d = width * 0.2 * intensity
        rect = np.array([
            [random.uniform(-d, d), random.uniform(-d, d)],
            [width + random.uniform(-d, d), random.uniform(-d, d)],
            [width + random.uniform(-d, d), height + random.uniform(-d, d)],
            [random.uniform(-d, d), height + random.uniform(-d, d)]], dtype = "float32")
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        img = cv2.warpPerspective(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)

        mutations.append(img)
        
    return mutations

def flip_create(img, annotation):
    classid = int(annotation[0])
    x1 = annotation[3]
    y1 = annotation[4]
    x2 = annotation[5]
    y2 = annotation[6]
    width, height = img.shape

    # List of classid of axial symmetrical signs
    flip_h = np.array([11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
    flip_v = np.array([1, 5, 12, 15, 17])
    # List of classid of rotationally symmetrical signs
    rotate = np.array([32, 40])
    # List of pair of classid that can be flipped horizontally and re-classified
    flip_reclass = np.array([
        [19, 20], [20, 19],
        [33, 34], [34, 33],
        [36, 37], [37, 36],
        [38, 39], [39, 38],
    ])

    newset = [[classid, img]] # The set includes original image

    # Flip to create more variations
    if classid in flip_h:
        newset.append([classid, cv2.flip(img, 1)])
    if classid in flip_v:
        newset.append([classid, cv2.flip(img, 0)])
    if classid in rotate:
        newset.append([classid, cv2.flip(img, -1)])
    if classid in flip_reclass[:, 0]:
        flip_class = flip_reclass[flip_reclass[:, 0] == classid][0][1]
        newset.append([flip_class, cv2.flip(img, 1)])
        
    return newset

def preprocess(img, annotation, ycrcb=False, noclahe=False):
    width, height, depth = img.shape

    # Convert to grayscale
    if ycrcb:
        cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    channels = cv2.split(img)
    img = channels[0]
    
    # Apply global histogram equalize than CLAHE, Contrast Limited Adaptive Histogram Equalization
    img = cv2.equalizeHist(img)
    if not noclahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)

    if annotation:    
        x1 = annotation[3]
        y1 = annotation[4]
        x2 = annotation[5]
        y2 = annotation[6]
        # Crop to roi
        rect = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]], dtype = "float32")
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        img = cv2.warpPerspective(img, M, (width, height))
    
    return img

def hash_str(text):
    h = hashlib.new('ripemd160')
    h.update(text.encode('utf-8'))
    return h.hexdigest()

def summarize(reset=False):
    summary = None
    pklpath = os.path.normpath(os.path.join(tempfile.gettempdir(), hash_str(ARGS.src)+'.pkl'))

    if os.path.exists(pklpath) and not reset: # Load from pickle
        with open(pklpath, 'rb') as input:
            summary = pickle.load(input)
    if not summary: # Scan the directory
        summary = {
            'classcount' : {},
            'total' : 0,
        }
        print('Parsing annotations from source directory...', ARGS.src)
        pathlist = Path(ARGS.src).glob('**/*.csv')
        for path in pathlist:
            # A image folder with .csv file (annotations)
            files = {}
            path_in_str = str(path)
            dir_this = os.path.split(path_in_str)[0]
            with open(path_in_str, newline='') as csvfile:
                colnames = None
                for item in csv.reader(csvfile, delimiter=';'):
                    # Parse annotations
                    if colnames:
                        #if re.match('.+\.ppm', item[0], re.IGNORECASE):
                        filename = item[colnames['filename']]
                        if 'classid' in colnames:
                            classid = int(item[colnames['classid']])
                        else:
                            classid = 0 # Unknown sample (test dataset)
                        summary['total'] += 1
                        if classid in summary['classcount']:
                            summary['classcount'][classid] += 1
                        else:
                            summary['classcount'][classid] = 1
                    else:
                        if re.match('filename', item[0], re.IGNORECASE):
                            colnames = {}
                            for k, v in enumerate(item):
                                colnames[v.lower()] = k
                            logging.debug('Column names parsed from a .csv file')
                            logging.debug(colnames)
        with open(pklpath, 'wb') as output:
            pickle.dump(summary, output, pickle.HIGHEST_PROTOCOL)

    return summary

def create_empty_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        # Empty target directory
        pathlist = Path(directory).glob('**/*')
        for path in pathlist:
            path_in_str = str(path)
            if os.path.isfile(path_in_str):
                os.unlink(path_in_str)

def parse_annotations(directory, is_test, dir_dest, thumbnail_dir=''):
    # Parse training data annotations
    num_processed = 0
    filename_seq = dict()

    thumbnails = dict()

    pathlist = Path(directory).glob('**/*.csv')
    for path in pathlist:
        path_in_str = str(path)
        dir_this = os.path.split(path_in_str)[0]
        with open(path_in_str, newline='') as csvfile:
            print()
            print('Parsing annotation', os.path.abspath(path_in_str), '...')
            colnames = None
            for row in csv.reader(csvfile, delimiter=';'):
                if colnames: # Column names already parsed
                    filename = row[colnames['filename']]
                    if 'classid' in colnames:
                        classid = int(row[colnames['classid']])
                    else:
                        classid = 0 # Unknown sample (test dataset)

                    width = int(row[colnames['width']])
                    height = int(row[colnames['height']])
                    x1 = int(row[colnames['roi.x1']])
                    y1 = int(row[colnames['roi.y1']])
                    x2 = int(row[colnames['roi.x2']])
                    y2 = int(row[colnames['roi.y2']])
                    annotation = (classid, width, height, x1, y1, x2, y2)

                    # Main operations: read, process, write
                    inpath = os.path.join(dir_this, filename)
                    if os.path.exists(inpath):
                        img = cv2.imread(inpath, 1)

                        # Save thumbnail for preview
                        width, height, depth = img.shape
                        if min(width, height) > min(thumbnails[classid]):
                            cv2.imwrite(os.path.join(thumbnail_dir, str(classid)+'.jpg'), img)
                            thumbnails[classid] = [width, height]

                        img = preprocess(img, annotation, ycrcb=ARGS.ycrcb, noclahe=ARGS.noclahe)

                        outpath = ''
                        if is_test:
                            newset = [[annotation[0], img]]
                        else:
                            # Flip or rotate to expand dataset
                            newset = flip_create(img, annotation)

                        for item_class, item_img in newset:
                            outdir = os.path.normpath(os.path.join(dir_dest, str(item_class)))
                            if item_class not in filename_seq:
                                if not is_test: print('Creating class directory: ', os.path.abspath(outdir))
                                filename_seq[item_class] = 0
                                create_empty_directory(outdir)

                            if is_test:
                                outname = filename
                            else:
                                #outname = str(item_class).zfill(4) + '-' + str(filename_seq[item_class]).zfill(6) + '.ppm'
                                outname = str(uuid4()) + '.ppm' # Randomized names effectively randomize file order
                                filename_seq[item_class] += 1

                            outpath = os.path.normpath(os.path.join(outdir, outname))
                            item_img = cv2.resize(item_img, (ARGS.dimension, ARGS.dimension)) # Resize to target dimension and save
                            retval = cv2.imwrite(outpath, item_img)
                            if not retval:
                                print('Error imwrite():', os.path.abspath(outpath))

                        num_processed += 1
                        if num_processed%2048==0:
                            print(str(num_processed), 'image files processed...')
                    else:
                        print('Skipped missing image file:', filename)

                else:
                    # Create table of column-name to dict-index
                    if re.match('filename', row[0], re.IGNORECASE):
                        colnames = {}
                        for k, v in enumerate(row):
                            colnames[v.lower()] = k
                    else:
                        print('First row has no column names. Skipped this annotations file.')

def main():
    # Download and unzip dataset and annotations
    links_download = [
        ['training', 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip', 'GTSRB/Final_Training/Images'],
        ['test', 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip', 'GTSRB/Final_Test/Images'],
        ['test_annotations', 'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip', 'GTSRB/Final_Test/Images/GT-final_test.csv'],
    ]
    train_dir = os.path.normpath(links_download[0][2])
    test_dir = os.path.normpath(links_download[1][2])
    data_dir = os.path.normpath(ARGS.data_dir)
    if not os.path.exists(data_dir):
        print()
        print('Directory', os.path.abspath(data_dir), 'created.')
        os.makedirs(data_dir)
    for download_name, download_url, extracted in links_download:
        head, tail = os.path.split(download_url)
        filename_pure = tail
        # Download...
        path_local = os.path.normpath(os.path.join(data_dir, filename_pure))
        if not os.path.exists(path_local):
            print()
            print('Downloading', filename_pure, 'to', path_local)
            wget.download(download_url, path_local)
        # Unzip...
        path_extracted = os.path.normpath(os.path.join(data_dir, extracted))
        if not os.path.exists(path_extracted):
            with zipfile.ZipFile(path_local, 'r') as zip_ref:
                if download_name=='test_annotations':
                    # Put additional annotations for test in data directory and remove annotations without classid
                    pathlist = Path(test_dir).glob('**/*.csv')
                    for path in pathlist:
                        path_in_str = str(path)
                        if os.path.isfile(path_in_str): os.unlink(path_in_str)
                    dir_extract = os.path.normpath(os.path.join(data_dir, test_dir))
                else:
                    dir_extract = data_dir
                print()
                print('Extracting', filename_pure, 'to', os.path.abspath(dir_extract))
                zip_ref.extractall(dir_extract)

    train_dir = os.path.normpath(os.path.join(data_dir, train_dir))
    test_dir = os.path.normpath(os.path.join(data_dir, test_dir))
    parse_annotations(train_dir, is_test=False,
        dir_dest=os.path.normpath(os.path.join(ARGS.dest, 'train')),
        thumbnail_dir=os.path.normpath(os.path.join(ARGS.dest, 'thumbnail')))
    parse_annotations(test_dir, is_test=True, dir_dest=os.path.normpath(os.path.join(ARGS.dest, 'test')))

    print()
    print('done')
    print()

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Clean up images and transform to generate more samples""")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../../data',
        help='Path to main directory of downloaded and processed image samples.'
    )
    parser.add_argument(
        '--dest',
        type=str,
        default='../../data/GTSRB/processed',
        help='Path to destination directory of processed images.'
    )
    parser.add_argument(
        '--noclahe',
        action='store_true',
        help='Do not use CLAHE for equalization.'
    )
    parser.add_argument(
        '--ycrcb',
        action='store_true',
        help='Use Y channel of YCrCb instead of luminance.'
    )
    parser.add_argument(
        '--dimension',
        type=int,
        default=32,
        help='Target dimension of prepared samples.'
    )
    ARGS, unknown = parser.parse_known_args()
    logging.debug('Processing images...')
    logging.debug(ARGS)
    if unknown:
        print(unknown)
        raise Exception('Unknown argument')
    main()

