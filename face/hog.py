import pydoc
import cv2
import math
import numpy as np

import argparse
from getdata import recursive_file_iterator

ARGS = None

def calc_num_of_blocks(map_size, block_size, block_stride):
    return [(map_size[0]-block_size[0])//block_stride[0]+1, (map_size[1]-block_size[1])//block_stride[1]+1]

def load_image(path, wnd):
    img = cv2.imread(path)
    #img = cv2.resize(img, (64, 64))
    width, height, depth = img.shape
    print('sample dimension', width, height, width*height)

    win_size = (64, 64) # Decrease length of output
    block_size = (16, 16) # In pixels
    block_stride = (8, 8) # In pixels
    cell_size = (8, 8) # In pixels
    nbins = 9
    deriv_aperture = 1
    win_sigma = 4.
    histogram_norm_type = 0
    threshold = 2.0000000000000001e-01
    gamma_correction = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size,
        nbins, deriv_aperture, win_sigma, histogram_norm_type, threshold, gamma_correction, nlevels)
    hog.save("hog.xml")

    win_stride = (8, 8) # Not affecting length of output
    padding = (8, 8)
    locations = ((10, 20),)
    hist = hog.compute(img, win_stride, padding, locations) # Supply locations to compute histograms of a single window

    hog_width = ((width + padding[0]*2) - win_size[0])//win_stride[0]*win_stride[0] + win_size[0]
    hog_height = ((height + padding[1]*2) - win_size[1])//win_stride[1]*win_stride[1] + win_size[1]
    print('padded image', hog_width, hog_height)

    # For the image...
    print('win_size', win_size)
    print('win_stride', win_stride)
    win_dim = calc_num_of_blocks((hog_width, hog_height), win_size, win_stride)
    print('win dimension', win_dim, win_dim[0]*win_dim[1])

    # For each window...
    win_size_p = ((win_size[0] - win_size[0])//block_stride[0]*block_stride[0] + win_size[0],
        (win_size[1] - win_size[1])//block_stride[1]*block_stride[1] + win_size[1])
    print('padded window', win_size_p)
    block_dim = calc_num_of_blocks(win_size_p, block_size, block_stride)
    print('blocks in window', block_dim, block_dim[0]*block_dim[1])
    cell_dim = (block_size[0]//cell_size[0], block_size[1]//cell_size[1])
    print('cells in block', cell_dim, cell_dim[0]*cell_dim[1])

    print('num of hist.', block_dim[0]*block_dim[1]*cell_dim[0]*cell_dim[1])

    # Visualize blocks
    for p1 in locations:
        p2 = np.add(p1, win_size)
        cv2.rectangle(img, p1, tuple(p2), (128, 255, 128), 1)
    
    print('block_stride', hog_width//block_stride[0], hog_height//block_stride[1])
    print('blocks', hog_width//block_size[0], hog_height//block_size[1])
    print('cells', hog_width//cell_size[0], hog_height//cell_size[1])
    print('num of bins', hist.shape[0])
    print('num of hist', hist.shape[0]/nbins)
    print('dim of blocks', math.sqrt(hist.shape[0]/nbins))

    hist = hog.compute(img, win_stride, padding) # Omit locations to compute whole image
    print()
    print('compute without locations')
    print('num of bins', hist.shape[0])
    print('num of hist', hist.shape[0]/nbins)
    print('num of hist.', win_dim[0]*block_dim[0]*cell_dim[0]*win_dim[1]*block_dim[1]*cell_dim[1])

    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    (rects, weights) = hog.detectMultiScale(img, winStride=(8, 8), padding=(16, 16), scale=1.05, useMeanshiftGrouping=False)
    print(rects)
    print(weights)

    cv2.resizeWindow(wnd, width, height)
    cv2.imshow(wnd, img)

def next_file(refite):
    for f in refite:
        print(f)
        load_image(f, 'wnd')
        break

def main():
    pydoc.writedoc("cv2.HOGDescriptor")

    refite = recursive_file_iterator(ARGS.source_dir)
    cv2.namedWindow('wnd', cv2.WINDOW_NORMAL)

    next_file(refite)

    while True:
        k = cv2.waitKeyEx(0)
        if k==27 or k==-1:
            break
        elif k in (2490368, 2424832):
            next_file(refite)
        elif k in (2621440, 2555904):
            next_file(refite)
        else:
            print('unregistered key_code:', k)

    cv2.destroyAllWindows()

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Clean up images and transform to generate more samples""")
    parser.add_argument(
        '--source_dir',
        type=str,
        default='../../data/face/lfw/extracted/lfw',
        help='Path to the data.'
    )
    ARGS, unknown = parser.parse_known_args()
    if unknown:
        print(unknown)
        raise Exception('Unknown argument')
    main()