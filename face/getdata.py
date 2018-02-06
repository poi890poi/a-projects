import cv2
import numpy as np
import scipy.stats

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
from threading import Thread

import logging
from logging.handlers import RotatingFileHandler
import pprint

logfile = os.path.normpath(os.path.join(tempfile.gettempdir(), 'imgxform.log'))
logging.basicConfig(filename=logfile, level=logging.WARNING)
log = logging.getLogger()
handler = RotatingFileHandler(logfile, maxBytes=512*1024, backupCount=1)
log.addHandler(handler)

ARGS = None

def hash_str(text):
    h = hashlib.new('ripemd160')
    h.update(text.encode('utf-8'))
    return h.hexdigest()

def norm_join_path(head, trail):
    return os.path.normpath(os.path.join(head, trail))

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

def recursive_file_iterator(directory):
    for entry in os.scandir(directory):
        if entry.is_dir():
            yield from recursive_file_iterator(entry.path)
        else:
            yield entry.path

class Dataset(Thread):
    def __init__(self, dir, source):
        Thread.__init__(self)
        self.source = source
        self.source_name = source[0]
        self.url = source[1]
        self.local_dir = prepare_dir(dir, self.source_name)
        self.extract_dir = norm_join_path(self.local_dir, 'extracted')
        components = os.path.split(self.url)
        self.filename = components[1]
        components = os.path.splitext(components[1])
        self.clearname = components[0]
        self.extension = components[1]
        self.archive_path = norm_join_path(dir, self.filename)
        self.file_iterators = recursive_file_iterator(self.extract_dir)

    def run(self):
        self.download()

        # Dataset specific actions
        if self.source_name=='vggface':
            self.vggface_download()

    def vggface_download(self):
        return
        # The dataset has no gender and age information
        # Complete this part later if to use this dataset
        path = self.next_file()
        while path and limit>0:
            with open(path, 'r') as f:
                line = f.readline()
                while line:
                    print(limit, line.split())
                    # id url left top right bottom pose detection_score curation
                    id = line[0]
                    url = line[1]
                    box = [line[2], line[3], line[4], line[5]]
                    pose = line[6]
                    # TODO: Download image...
                    line = f.readline()
                    break
            path = self.next_file()

    def download(self):
        print(self.archive_path)
        if not os.path.exists(self.archive_path):
            # Download...
            print()
            print('Downloading', self.filename, 'to', self.archive_path)
            wget.download(self.url, self.archive_path)
        if not os.path.exists(self.extract_dir):
            # Extracting
            print()
            print('Extracting', self.archive_path, 'to', self.extract_dir)
            self.extract_dir = prepare_dir(self.local_dir, 'extracted')
            shutil.unpack_archive(self.archive_path, self.extract_dir)

    def next_file(self):
        for f in self.file_iterators:
            return f

    @staticmethod
    def list_dataset():
        sources = [
            ['lfw', 'http://vis-www.cs.umass.edu/lfw/lfw.tgz', ''],
            ['vggface', 'http://www.robots.ox.ac.uk/~vgg/data/vgg_face/vgg_face_dataset.tar.gz', ''],
            #['imdb-face', 'https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar', ''],
            #['imdb-meta', 'https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_meta.tar', ''],
            ['wiki-face', 'https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki.tar.gz', ''],
        ]
        return sources


def main():
    # Download and unzip dataset and annotations
    threads = list()
    for source in Dataset.list_dataset():
        downloader = Dataset(ARGS.data_dir, source)
        downloader.start()
        threads.append(downloader)

    for t in threads:
        t.join()

    print()
    print('done')
    print()

    active_dataset = threads[0]
    print('active', active_dataset.extract_dir)

    # Invoke an inspector
    limit = 1
    f = active_dataset.next_file()
    while f and limit>0:
        print(f)
        f = active_dataset.next_file()
        limit -= 1


if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Download and extract datasets""")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../../data/face',
        help='Path to destination directory of downloaded and processed images.'
    )
    ARGS, unknown = parser.parse_known_args()
    if unknown:
        print(unknown)
        raise Exception('Unknown argument')
    main()

