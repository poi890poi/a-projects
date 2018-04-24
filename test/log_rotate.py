import logging, logging.handlers, sys

import cv2

from queue import Queue, Empty
import threading
import time
import random
import sys
import traceback

LOG_DIRECTORY = '../../log/'
formatter = logging.Formatter('[%(asctime)s] %(levelname)s (%(process)d) %(module)s %(lineno)d: %(message)s')

sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)

fh = logging.handlers.RotatingFileHandler(LOG_DIRECTORY + 'log_rotate_testing.log', maxBytes=4*1024*1024, backupCount=8)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

logger = logging.getLogger('ALogger')
logger.propagate = False
logger.setLevel(logging.DEBUG)
logger.addHandler(sh)
logger.addHandler(fh)

def my_handler(type, value, tb):
    logger.error('\n'.join(['Uncaught Exception'] + list(traceback.format_tb(tb, limit=32)) + [type.__name__+': '+str(value),]))

# Install exception handler
sys.excepthook = my_handler

debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical

class Worker(threading.Thread):
    def __init__(self, in_queue, out_queue):
        self.in_queue = in_queue
        self.out_queue = out_queue
        threading.Thread.__init__(self)
        self.is_crashed = False
        self.is_init = True

    def run(self):
        # Initialization...

        while True:
            if self.is_init:
                debug('*** INITIALIZING...')

                t_ = random.random()
                time.sleep(t_)
                self.is_init = False
                debug('*** INITIALIZED...')
            elif not self.is_crashed:
                try:
                    t_now = time.time()

                    t_enqueued = self.in_queue.get() # queue::get() is blocking by default

                    if t_now - t_enqueued > 0.5:
                        warning('Skip...'+str(t_now-t_enqueued))
                    else:
                        t_ = random.random() / 4
                        time.sleep(t_)
                        info('Do something...'+str(t_))

                    rand = random.randint(0, 16)
                    if rand == 4:
                        warning('Thread about to crash...')
                        raise RuntimeError('Thread crashed!')

                    # Signals to queue job is done
                    self.in_queue.task_done()
                except:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    error('\n'.join(['Thread exception: {}'.format(threading.current_thread())]+list(traceback.format_tb(exc_traceback, limit=5))))
                    self.is_crashed = True

            if self.is_crashed:
                self.in_queue.task_done()
                break

debug('Wait for initialization...')

in_queue = Queue()
out_queue = Queue()
threads = []

for i in range(16):
    t = Worker(in_queue, out_queue)
    t.setDaemon(True)
    t.start()
    threads.append(t)

debug('Initialization done...')

cv2.namedWindow('blocking', cv2.WINDOW_NORMAL)

while True:
    t_now = time.time()
    t_ = random.random() / 16
    time.sleep(t_)

    for i, t in enumerate(threads):
        if t.is_crashed:
            #del(threads[i])
            warning('Restarting thread... {}'.format(i))
            threads[i] = Worker(in_queue, out_queue)
            threads[i].setDaemon(True)
            threads[i].start()
        elif not t.is_init:
            if in_queue.qsize() <= 32:
                debug('Enqueue task... now: {}, qsize: {}'.format(t_now, in_queue.qsize(), t.isAlive(), t.is_init))
                in_queue.put(t_now)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

raise RuntimeError('Termination; test unhandled critical error logging')