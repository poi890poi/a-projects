import numpy as np
import cv2
import time
from datetime import datetime
import base64

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError
import urllib3

import argparse

from queue import Queue, Empty
import threading

in_queue = Queue()
out_queue = Queue()

class ThreadUrl(threading.Thread):
    """Threaded Url Grab"""
    def __init__(self, in_queue, out_queue):
        threading.Thread.__init__(self)
        self.in_queue = in_queue
        self.out_queue = out_queue
 
    def run(self):
        http = urllib3.PoolManager()
        while True:
            #grabs data from queue
            task = self.in_queue.get()

            if time.time() > task['timing']['t_expiration']:
                # Task waiting for too long in the queue; discard it.
                #print('skip frame')
                pass

            else:
                delay = time.time()*1000 - task['agent']['t_frame']
                print('frame to http request', delay, task['requests'][0]['services'][0]['mode'])

                postdata = json.dumps(task)
                try:
                    #self.response = None
                    #url = 'http://192.168.41.41:9000/predict'
                    url = 'http://10.129.11.4:9000/predict'
                    r = http.request('POST', url, body=postdata.encode())
                    response = json.loads(r.data.decode())
                    timing = response['timing']
                    server_time = timing['server_sent'] - timing['server_rcv']
                    total_time = (time.time() - timing['client_sent']) * 1000
                    client_time = total_time - server_time
                    #print(len(response['responses'][0]['services'][0]['results']['rectangles']))
                    response['agent'] = task['agent']
                    self.out_queue.put(response)
                    print('frame to http response', time.time()*1000 - task['agent']['t_frame'])
                    #print('response time:', total_time)
                    print()
                except URLError:
                    pass
                    
            #signals to queue job is done
            self.in_queue.task_done()

def time_string(input):
    input = int(input)
    milliseconds = str(input%1000).zfill(4)
    seconds = (input/1000) % 60
    seconds = str(int(seconds)).zfill(2)
    minutes = (input/(1000*60)) % 60
    minutes = str(int(minutes)).zfill(2)
    hours = (input/(1000*60*60)) % 24
    hours = str(int(hours)).zfill(2)

    return '{}:{}:{}.{}'.format(hours, minutes, seconds, milliseconds)

def main(args):
    name_table = {
        'd747': 'Natalie Portman',
        '05c1': 'Dana Carvey',
        'd364': 'Emma Watson',
        'e36e': 'Emma Watson',
        '3283': 'Taylor Swift',
        'ed29': "Conan O'Brien",
        'c584': "Harrison Ford",
        '62c6': "Graham Norton",
        '0d26': "Reese Witherspoon",
        '61fa': "Ryan Gosling",
        'f98c': "no one",
    }
    tests = {
        'cn01': {
            'video': '../../data/youtube_clips/conan/d_carvey_01.mp4',
        },
        'cn02': {
            'video': '../../data/youtube_clips/conan/j_daniels_01.mp4',
        },
        'cn03': {
            'video': '../../data/youtube_clips/conan/j_manganiello_01.mp4',
        },
        'cn04': {
            'video': '../../data/youtube_clips/conan/m_akerman_01.mp4',
        },
        'cn05': {
            'video': '../../data/youtube_clips/conan/n_leggero_01.mp4',
        },
        'np01': {
            'video': '../../data/youtube_clips/n_portman_01.mp4',
        },
        'ew01': {
            'video': '../../data/youtube_clips/e_watson_01.mp4',
        },
        'ts01': {
            'video': '../../data/youtube_clips/graham/t_swift_01.mp4',
        },
        'hf01': {
            'video': '../../data/youtube_clips/graham/h_ford_01.mp4',
        },
    }
    # Initialize video capture object
    if args.vfile:
        cap = cv2.VideoCapture(args.vfile)
    elif args.test:
        cap = cv2.VideoCapture(tests[args.test]['video'])
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print( length )
    else:
        cap = cv2.VideoCapture(0)

    interval = 1./25.
    t_ = time.time() + interval
    f_ = 0
    osd = []

    # Start the thread for URL fetching
    for i in range(1):
        t = ThreadUrl(in_queue, out_queue)
        t.setDaemon(True)
        t.start()

    EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    font = cv2.FONT_HERSHEY_SIMPLEX

    last_response = None
    mode = ''
    t_register_expiration = 0
    cd_play_ctrl = 0
    is_paused = False

    while(True):
        # Capture frame-by-frame
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                print('No more frame')
                break
            cached = np.array(frame)
        else:
            frame = np.array(cached)

        if frame is None:
            continue
        t_frame = time.time() * 1000

        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        f_ += 1
        if time.time() > t_:
            # Do something every <interval> seconds

            retval, bin_data = cv2.imencode('.jpg', frame)
            requests = {
                "requests": [
                    {
                        "requestId": "3287de74a3c742ccc81e80eab881eaec",
                        "media": {
                            "content": base64.b64encode(bin_data).decode()
                        },
                        "services": [
                            {
                                "type": "face",
                                #"model": "a-emoc", # Request emotion recognition
                                "model": "fnet",
                                "mode": mode,
                                "options": {
                                    "res_cap": 448,
                                    "factor": 0.6,
                                    "interp": "NEAREST",
                                    "fnet-cooldown": 1500,
                                }
                            }
                        ]
                    }
                ],
                'agent': {
                    'agentId': '192.168.41.41',
                    't_frame': t_frame,
                    'debug': 1,
                },
                'timing': {
                    'client_sent': time.time(),
                    't_expiration': time.time() + 0.25,
                }
            }

            if in_queue.qsize() >= 1:
                # in_queue is full
                pass
            else:
                in_queue.put(requests)
                #print('qsize', in_queue.qsize())

            f_ = 0
            t_ = time.time() + interval

        try:
            last_response = out_queue.get_nowait()
            print('frame to get_nowait()', time.time()*1000 - last_response['agent']['t_frame'])
        except Empty:
            pass

        if last_response:
            #print(len(t.response['responses'][0]['services'][0]['results']['rectangles']))
            for r in last_response['responses']:
                for service in r['services']:
                    for i, r in enumerate(service['results']['rectangles']):
                        cv2.rectangle(frame, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (255, 255, 0))
                        if 'emotions' in service['results'] and len(service['results']['emotions']):
                            cv2.putText(frame, EMOTIONS[np.argmax(service['results']['emotions'][i])], (r[0], r[1]), font, 0.7, (255, 0, 255), 1, cv2.LINE_AA)
                        if 'identities' in service['results']:
                            identity = service['results']['identities']
                            name = identity['name'][i][0:4]
                            if name in name_table: name = name_table[name]
                            if not name: name = 'UNKNOWN'
                            confidence = identity['confidence'][i]
                            if confidence >= 0.3:
                                confidence = str(round(confidence, 2))
                                tag = '{} / {}'.format(name, confidence)
                                cv2.putText(frame, tag, (r[0], r[1]), font, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
                            else:
                                s_index = str(round(service['results']['sort_index'][str(i)], 2))
                                tag = '{}'.format(s_index)
                                cv2.putText(frame, tag, (r[0], r[1]), font, 0.7, (255, 255, 0), 1, cv2.LINE_AA)

        pos = cap.get(cv2.CAP_PROP_POS_MSEC)
        pos_str = time_string(pos)
        osd = [pos_str, repr(frame.shape)]
        if mode=='register':
            osd.append('REGISTERING NEW FACE...')
        if is_paused:
            osd.append('PAUSED')
        y_ = 30
        for line in osd:
            cv2.putText(frame, line, (10, y_), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
            y_ += 36

        if time.time() > t_register_expiration:
            mode = ''

        # Display the resulting frame
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            print('QQQQQQQ')
            break
        elif key==ord('r'):
            mode = 'register'
            t_register_expiration = time.time() + 0.5
        elif key==ord('.'): # Forward jump
            for i in range(25*30):
                cap.grab()
        elif key==ord(','): # Back jump
            pass
        elif key==ord(' '): # Pause
            if time.time() > cd_play_ctrl: # De-bounce of play control
                cd_play_ctrl = time.time() + 0.25
                if is_paused:
                    is_paused = False
                else:
                    is_paused = True

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Feeding video frames to computer vision server""")
    parser.add_argument(
        '--vfile',
        type=str,
        default='',
        help='Path to the media file.'
    )
    parser.add_argument(
        '--test',
        type=str,
        default='',
        help='Name of test preset.'
    )
    ARGS, unknown = parser.parse_known_args()
    if unknown:
        print(unknown)
        raise Exception('Unknown argument')
    main(ARGS)