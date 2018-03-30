import cv2
import numpy as np
import base64
import uuid
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import os.path

from shared.utilities import *
from facer.train import *
from facer.datagen import *
from facer.predict import *
from server.server import *
from emotion.emotion_recognition import fxpress, fxpress_train
from facer.emotion import EmotionClassifier
from facer.face_app import FaceApplications

def main():
    if ARGS.test=='train':
        train(ARGS)
    elif ARGS.test=='gen':
        gen(ARGS)
    elif ARGS.test=='predict':
        predict(ARGS)
    elif ARGS.test=='server':
        server_start(ARGS)
    elif ARGS.test=='server_production':
        server_start(ARGS, ARGS.port)
    elif ARGS.test=='hnm':
        hnm(ARGS)
    elif ARGS.test=='val':
        val(ARGS)
    elif ARGS.test=='fer':
        fer(ARGS)
    elif ARGS.test=='fxpress':
        fxpress(ARGS)
    elif ARGS.test=='fxpress_train':
        fxpress_train(ARGS)
    elif ARGS.test=='emoc':
        classifier = EmotionClassifier()
        classifier.build_network(ARGS)
        classifier.val(ARGS)
    elif ARGS.test=='face_app':
        face_app = FaceApplications()
        face_app.detect()
    elif ARGS.test=='face_benchmark':
        interpolations = ['NEAREST', 'LINEAR', 'AREA']
        resolutions = [256, 320, 384, 448, 512, 640, 1024, 1280]
        factors = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        expectations = [
            ['0001.jpg', 4],
            ['0002.jpg', 1],
            ['0003.jpg', 1],
            ['0004.jpg', 0],
            ['0005.jpg', 1],
            ['0006.jpg', 1],
            ['0007.jpg', 59],
            ['0008.jpg', 6],
            ['0009.jpg', 1],
            ['0010.jpg', 5],
            ['0011.jpg', 4],
            ['0012.jpg', 17],
            ['0013.jpg', 20],
            ['0014.jpg', 48],
            ['0015.jpg', 22],
        ]
        
        log_file = open('../data/face_benchmark.csv', 'w')
        log_file.write('time,precision_index,options\n')

        for interp in interpolations:
            for res_cap in resolutions:
                for factor in factors:
                    #res_cap = resolutions[0]
                    #factor = factors[0]
                    #interp = interpolations[0]
                    options = {
                        'res_cap': res_cap,
                        'factor': factor,
                        'interp': interp,
                    }
                    if interp=='NEAREST':
                        interpolation = cv2.INTER_NEAREST
                    elif interp=='LINEAR':
                        interpolation = cv2.INTER_LINEAR
                    elif interp=='AREA':
                        interpolation = cv2.INTER_AREA

                    iterations = 10
                    file_count = len(expectations)
                    time_sampling = np.zeros((file_count, iterations,), dtype=np.float)
                    pi_sampling = np.zeros((file_count, iterations,), dtype=np.float)

                    image_dir = '../data/face_benchmark/'
                    for k, item in enumerate(expectations):
                        filename, expected_faces = item
                        inpath = image_dir + filename
                        img = cv2.imread(inpath, 1)
                        if img is None:
                            break

                        img, scaling = ImageUtilities.fit_resize(img, maxsize=(res_cap, res_cap), interpolation=interpolation)
                        retval, bindata = cv2.imencode('.jpg', img)
                        bindata_b64 = base64.b64encode(bindata).decode()

                        requests = {
                            'requests': [
                                {
                                    'requestId': str(uuid.uuid1()),
                                    'media': {
                                        'content': bindata_b64
                                    },
                                    'services': [
                                        {
                                            'type': 'face_',
                                            'model': 'alexnet-emoc',
                                            'options': options
                                        }
                                    ]
                                }
                            ],
                            'timing': {
                                'client_sent': time.time()
                            }
                        }

                        for i in range(iterations):
                            #url = 'http://10.129.11.4/cgi/predict'
                            requests['timing']['client_sent'] = time.time()
                            url = 'http://192.168.41.41:8080/predict'
                            postdata = json.dumps(requests)
                            #print()
                            #print(postdata)
                            #print()
                            request = Request(url, data=postdata.encode())
                            response = json.loads(urlopen(request).read().decode())
                            timing = response['timing']
                            server_time = timing['server_sent'] - timing['server_rcv']
                            #print('server time:', server_time)
                            total_time = (time.time() - timing['client_sent']) * 1000
                            client_time = total_time - server_time
                            print('response time:', total_time)
                            pi = 0.
                            for r_ in response['requests']:
                                for s_ in r_['services']:
                                    rects_ = s_['results']['rectangles']
                                    if expected_faces:
                                        pi = len(rects_) / expected_faces
                                    elif len(rects_):
                                        pi = expected_faces / len(rects_)
                                    else:
                                        pi = 1.0
                                    #print('faces detected:', len(rects_), pi)
                            time_sampling[k][i] = total_time
                            pi_sampling[k][i] = pi

                            #time.sleep(0.5)

                            #print()
                            #print(response)
                            #print()

                    time_mean = np.mean(time_sampling)
                    pi_mean = np.mean(pi_sampling) * 100
                    print(time_mean, pi_mean)
                    print()
                    log_file.write(','.join([str(time_mean), str(pi_mean), json.dumps(options)])+'\n')
                    log_file.flush()

        log_file.close()


if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Test units for face applications""")
    parser.add_argument(
        '--test',
        type=str,
        default='',
        help='Name of the test unit to run.'
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Display window and wait for user input, for preview.'
    )
    parser.add_argument(
        '--nolog',
        action='store_true',
        help='Skip logging to save disk space during training.'
    )
    parser.add_argument(
        '--subset',
        type=str,
        default='',
        help='Name of subset of dataset to be processed.'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=0,
        help='Count of entries to be processed or generated.'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port number of web server to listen to.'
    )
    parser.add_argument(
        '--model',
        type=str,
        #default='./server/models/12-net/model.ckpt',
        default='../models/cascade/checkpoint/model.ckpt',
        help='Path and prefix of Tensorflow checkpoint.'
    )
    parser.add_argument(
        '--cascade',
        type=int,
        default=12,
        help='Level of cascade CNN '
    )
    parser.add_argument(
        '--annotations',
        type=str,
        default='../data/face/wiki-face/extracted/wiki/wiki.mat',
        help='Path to annotations.'
    )
    parser.add_argument(
        '--train_id',
        type=str,
        default='',
    )
    ARGS, unknown = parser.parse_known_args()
    if unknown:
        print(unknown)
        raise Exception('Unknown argument')
    main()