"""
These codes test the simple behavior of Python threading and synchronized queue
"""

from queue import Queue, Empty
import threading
import time
import random
import cv2

class Worker(threading.Thread):
    def __init__(self, in_queue, out_queue):
        self.in_queue = in_queue
        self.out_queue = out_queue
        threading.Thread.__init__(self)

    def run(self):
        while True:
            #grabs data from queue
            task = self.in_queue.get() # queue::get() is blocking by default

            print(threading.current_thread(), task)
            
            # Pretend doing a random task that takes random time t_
            t_ = random.random()
            time.sleep(t_)

            # Pretend that the task is done
            task['out_queue'].put({
                'requester': task['thread'],
                'state': 'FINISHING',
                'elapsed': time.time()*1000 - task['time'],
            })

            #signals to queue job is done
            self.in_queue.task_done()

class Requester(threading.Thread):
    def __init__(self):
        self.out_queue = Queue()
        threading.Thread.__init__(self)

    def run(self):
        while True:
            t_ = random.random()
            print(threading.current_thread(), t_)
            time.sleep(t_)

            MainThread().do_something({
                'thread': threading.current_thread(),
                'out_queue': self.out_queue,
                'state': 'REQUEST',
                'time': time.time()*1000,
            })

            result = self.out_queue.get()
            if result['requester']==threading.current_thread():
                print('matched', result)
            else:
                print('not matched', threading.current_thread(), result)
                raise

            self.out_queue.task_done()

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class MainThread(metaclass=Singleton):
    def __init__(self):
        self.in_queue = Queue()
        self.out_queue = Queue()
        for i in range(5):
            t = Requester()
            t.setDaemon(True)
            t.start()
        for i in range(11):
            t = Worker(self.in_queue, self.out_queue)
            t.setDaemon(True)
            t.start()

    def do_something(self, task):
        MainThread().in_queue.put(task)

mt = MainThread()

cv2.namedWindow('blocking', cv2.WINDOW_NORMAL)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


