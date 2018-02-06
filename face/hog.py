import pydoc
import cv2
import math
import numpy as np

import argparse
import collections

from getdata import recursive_file_iterator

ARGS = None

def calc_num_of_blocks(map_size, block_size, block_stride):
    return [(map_size[0]-block_size[0])//block_stride[0]+1, (map_size[1]-block_size[1])//block_stride[1]+1]

def hog_compute(img, p):
    width, height, depth = img.shape
    
    win_size = p.w # Decrease length of output
    block_size = p.b # In pixels
    block_stride = p.b_stride # In pixels
    cell_size = p.c # In pixels
    nbins = p.nbins
    deriv_aperture = p.aperture
    win_sigma = p.sigma
    histogram_norm_type = p.norm
    threshold = p.t
    gamma_correction = p.g
    nlevels = p.nlevels
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size,
        nbins, deriv_aperture, win_sigma, histogram_norm_type, threshold, gamma_correction, nlevels)
    hog.save("hog.xml")

    win_stride = p.w_stride # Not affecting length of output
    padding = p.padding

    hist = hog.compute(img, win_stride, padding) # Omit locations to compute whole image
    return hist

    # Visualize blocks
    """for y in range(block_dim[1]):
        for x in range(block_dim[2]):
        p2 = np.add(p1, win_size)
        cv2.rectangle(img, p1, tuple(p2), (128, 255, 128), 1)"""

    """hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    (rects, weights) = hog.detectMultiScale(img, winStride=(8, 8), padding=(16, 16), scale=1.05, useMeanshiftGrouping=False)
    print(rects)
    print(weights)"""


def load_image(path, wnd):
    img = cv2.imread(path)
    #img = cv2.resize(img, (64, 64))
    width, height, depth = img.shape

    cv2.resizeWindow(wnd, width, height)

    return img

def next_file(refite):
    for f in refite:
        print(f)
        return load_image(f, 'wnd')

def draw(background, hog_p, wnd, cursor, hist):
    img = np.copy(background)
    width, height, depth = img.shape

    i_size = np.array((width, height))
    win_size = np.array(hog_p.w) # Decrease length of output
    win_stride = np.array(hog_p.w_stride) # Not affecting length of output
    padding = np.array(hog_p.padding)

    block_size = np.array(hog_p.b) # In pixels
    block_stride = np.array(hog_p.b_stride) # In pixels
    cell_size = np.array(hog_p.c) # In pixels
    nbins = np.array(hog_p.nbins)

    #print('i_size', i_size)
    scan_size = (i_size + padding * 2 - win_size) // win_stride * win_stride + win_size
    #print('scan_size', scan_size)
    win_dim = np.array(calc_num_of_blocks(scan_size, win_size, win_stride))
    #print('win_dim', win_dim)
    block_dim = np.array(calc_num_of_blocks(win_size, block_size, block_stride))
    #print('block_dim', block_dim)
    cell_dim = block_size // cell_size
    #print('cells in block', cell_dim, cell_dim[0]*cell_dim[1])
    #print('num of hist', hist.shape[0]/nbins)
    #print('num of hist.', win_dim[0]*block_dim[0]*cell_dim[0]*win_dim[1]*block_dim[1]*cell_dim[1])
    shape = win_dim
    shape = np.append(shape, block_dim)
    shape = np.append(shape, cell_dim)
    shape = np.append(shape, [nbins,])
    #print('num of bins', hist.shape[0])
    #print('num of calc. bins', np.prod(shape))
    print('shape', shape)
    hist = np.reshape(np.array(hist), shape)
    #print(hist)
    #return
    win = hist[cursor[0]][cursor[1]]
    print(win.shape)
    print(win[0][0])
    print(win[0][1])

    p1 = np.array(cursor * win_stride - padding)
    p2 = p1 + win_size
    cdim = win_size // cell_size

    len = cell_size // 2 - [2, 2]
    
    c1 = np.array(p1)
    for y in range(cdim[1]):
        for x in range(cdim[0]):
            c2 = tuple(np.add(c1, hog_p.c))
            cv2.rectangle(img, tuple(c1), c2, (255, 0, 255), 1)
            c = (c1 + c2) // 2
            cv2.line(img, tuple(c - len), tuple(c + len), (0, 255, 0), 1)
            c1[0] += hog_p.c[0]
        c1[0] = p1[0]
        c1[1] += hog_p.c[1]
    cv2.rectangle(img, tuple(p1), tuple(p2), (0, 255, 0), 1)
    
    cv2.imshow(wnd, img)

def main():
    pydoc.writedoc("cv2.HOGDescriptor")

    HogParameters = collections.namedtuple('HogParameters', 'w, b, b_stride, c, nbins, aperture, sigma, norm, t, g, nlevels, w_stride, padding')
    hog_p = HogParameters(
        w = (64, 64), # Window size, in pixels. default [64,128]
        b = (16, 16), # Block size, in pixels. default [16,16]
        b_stride = (8, 8), # Block stride, in pixels. default [8,8]
        c = (8, 8), # Cell size, in pixels. default [8,8]
        nbins = 9, # Number of bins. default 9
        aperture = 1, # aperture_size Size of the extended Sobel kernel, must be 1, 3, 5 or 7. default 1
        sigma = 4., # Windows sigma. Gaussian smoothing window parameter. default -1
        norm = 0, # Histogram normalization method. default 'L2Hys'
        t = 2.0000000000000001e-01, # L2 Hysterisis threshold. normalization method shrinkage. default 0.2
        g = 0, # Flag to specify whether the gamma correction preprocessing is required or not. default true
        nlevels = 64, # Maximum number of detection window increases. default 64
        w_stride = (8, 8), # Window stride, in pixels
        padding = (8, 8), # Padding of source image for window (not for block nor cell)
    )

    refite = recursive_file_iterator(ARGS.source_dir)
    cv2.namedWindow('wnd', cv2.WINDOW_NORMAL)

    img_original = next_file(refite)
    width, height, depth = img_original.shape

    i_size = np.array((width, height))
    win_size = np.array(hog_p.w) # Decrease length of output
    win_stride = np.array(hog_p.w_stride) # Not affecting length of output
    padding = np.array(hog_p.padding)
    scan_size = (i_size + padding * 2 - win_size) // win_stride * win_stride + win_size
    win_dim = np.array(calc_num_of_blocks(scan_size, win_size, win_stride))

    hist = hog_compute(img_original, hog_p)

    cursor = [0, 0]
    draw(img_original, hog_p, 'wnd', cursor, hist)

    while True:
        k = cv2.waitKeyEx(0)
        if k in (1, 27, -1): # esc
            break
        elif k in (13, 32): # enter, space
            img_original = next_file(refite)
            width, height, depth = img_original.shape
            hist = hog_compute(img_original, hog_p)
            cursor = [0, 0]
            p1 = (-hog_p.padding[0], -hog_p.padding[1])
            draw(img_original, hog_p, 'wnd', cursor, hist)
        elif k in (2490368, ): # up
            if cursor[1] > 0:
                cursor[1] -= 1
            draw(img_original, hog_p, 'wnd', cursor, hist)
        elif k in (2621440, ): # down
            if cursor[1] < win_dim[1] - 1:
                cursor[1] += 1
            draw(img_original, hog_p, 'wnd', cursor, hist)
        elif k in (2424832,): # left
            if cursor[0] > 0:
                cursor[0] -= 1
            draw(img_original, hog_p, 'wnd', cursor, hist)
        elif k in (2555904,): # right
            if cursor[0] < win_dim[0] - 1:
                cursor[0] += 1
            draw(img_original, hog_p, 'wnd', cursor, hist)
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