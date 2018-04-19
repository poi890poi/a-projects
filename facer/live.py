import numpy as np
import cv2
import time
import base64

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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
        while True:
            #grabs data from queue
            task = self.in_queue.get()

            if time.time() > task['timing']['t_expiration']:
                # Task waiting for too long in the queue; discard it.
                print('skip frame')

            else:
                print('frame to http request', time.time()*1000 - task['agent']['t_frame'])
                
                postdata = json.dumps(task)
            
                #self.response = None
                url = 'http://192.168.41.41:9000/predict'
                request = Request(url, data=postdata.encode())
                response = json.loads(urlopen(request).read().decode())
                timing = response['timing']
                server_time = timing['server_sent'] - timing['server_rcv']
                total_time = (time.time() - timing['client_sent']) * 1000
                client_time = total_time - server_time
                #print(len(response['responses'][0]['services'][0]['results']['rectangles']))
                response['agent'] = task['agent']
                self.out_queue.put(response)
                print('frame to http response', time.time()*1000 - task['agent']['t_frame'])
                print('response time:', total_time)
            
            print()
                    
            #signals to queue job is done
            self.in_queue.task_done()

# Initialize video capture object
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

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
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
            print('qsize', in_queue.qsize())

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
                    cv2.rectangle(frame, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (255, 0, 255))
                    if 'emotions' in service['results'] and len(service['results']['emotions']):
                        cv2.putText(frame, EMOTIONS[np.argmax(service['results']['emotions'][i])], (r[0], r[1]), font, 0.7, (255, 0, 255), 1, cv2.LINE_AA)
                    if 'identities' in service['results']:
                        identity = service['results']['identities']
                        name = identity['name'][i]
                        confidence = identity['confidence'][i]
                        #confidence = service['results']['confidences'][i]
                        if confidence > 0:
                            tag = name + ' / ' + str(confidence/1000)
                            cv2.putText(frame, tag, (r[0], r[1]), font, 0.7, (255, 0, 255), 1, cv2.LINE_AA)

    osd = [str(time.time()*1000), repr(frame.shape)]
    y_ = 30
    for line in osd:
        cv2.putText(frame, line, (10, y_), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
        y_ += 36

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()