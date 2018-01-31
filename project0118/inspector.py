import cv2
import numpy as np
import scipy.stats

import argparse
from pathlib import Path
import os.path

def append_candidate(img, candidates, entropy):
    width, height = img.shape
    hist = np.histogram(img, bins=16)
    #stats = scipy.stats.entropy(np.reshape(img, (width*height)))
    #stats = scipy.stats.entropy(hist[0])
    stats = scipy.stats.entropy(hist[0])
    candidates.append(img)
    entropy.append(stats)

def load_image(filelist, index):
    filename = filelist[index]
    img = cv2.imread(filename)
    print(filename)
    img = cv2.resize(img, (32, 32))

    #img = cv2.fastNlMeansDenoisingColored(img)

    width, height, depth = img.shape

    candidates = list()
    entropy = list()
    #candidate_names = ['y', 'cr', 'cb', 'hue', 'saturation', 'value']
    candidate_names = ['y', 'cr', 'cb', 'lightness', 'alpha', 'beta']

    #for conversion in [cv2.COLOR_RGB2YCrCb, cv2.COLOR_RGB2HSV]:
    for conversion in [cv2.COLOR_RGB2YCrCb, cv2.COLOR_RGB2LAB]:
        buffer = img
        cv2.cvtColor(buffer, conversion)
        channels = cv2.split(buffer)
        for channel in channels:
            channel = cv2.equalizeHist(channel)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            channel = clahe.apply(channel)
            append_candidate(channel, candidates, entropy)

    print(candidate_names[np.argmax(entropy)])

    buffer = np.resize(img, (width*3, height, depth))
    buffer[:, :, :] = 0
    buffer[0:width, :, :] = img
    buffer[width:width*2, :, :] = np.moveaxis(np.stack((candidates[0],)*3), 0, -1)
    buffer[width*2:width*3, :, :] = np.moveaxis(np.stack((candidates[np.argmax(entropy)],)*3), 0, -1)

    # Apply global histogram equalize than CLAHE, Contrast Limited Adaptive Histogram Equalization
    
    factor = 2
    cv2.namedWindow('wnd', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('wnd', buffer.shape[1]*factor, buffer.shape[0]*factor)
    cv2.imshow('wnd', buffer)

def prev_img(filelist, index):
    index -= 1
    if index < 0: index = len(filelist) - 1
    return index

def next_img(filelist, index):
    index += 1
    if index >= len(filelist): index = 0
    return index

def main():
    
    difficult_list = [
        '12566',
        '12600',
        '12606',
        '12524',
        '12506',
        '12505',
        '12503',
        '12500',
        '12473',
        '12456',
        '12452',
        '12432',
        '12411',
        '12399',
        '12380'
    ]

    filelist = list()
    dir = ARGS.dir
    #for f in difficult_list:
    #    filelist.append(os.path.normpath(os.path.join(dir, f + '.ppm')))
    pathlist = Path(dir).glob('**/*.ppm')
    for path in pathlist:
        path_in_str = str(path)
        filelist.append(path_in_str)

    index = 0
    load_image(filelist, index)

    while True:
        k = cv2.waitKeyEx(0)
        if k==27 or k==-1:
            break
        elif k in (2490368, 2424832):
            index = prev_img(filelist, index)
            load_image(filelist, index)
        elif k in (2621440, 2555904):
            index = next_img(filelist, index)
            load_image(filelist, index)
        else:
            print('unregistered key_code:', k)

    cv2.destroyAllWindows()

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Image Inspector""")
    parser.add_argument(
        '--dir',
        type=str,
        default='../../data/GTSRB/Final_Test/Images',
        help='Path to the directory of images to be inspected.'
    )
    ARGS, unknown = parser.parse_known_args()
    if unknown:
        print(unknown)
        raise Exception('Unknown argument')
    main()