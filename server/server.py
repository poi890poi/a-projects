import tornado.ioloop
import tornado.web
import tornado.template

import time
from uuid import uuid4
import os.path
from pathlib import Path
import pickle
import datetime
import json
import base64
import cv2
import numpy as np

import sys
import threading
import copy

from facer.predict import FaceClassifier
from facer.face_app import FaceApplications

from queue import Queue, Empty

from shared.utilities import NumpyEncoder
from shared.alogger import *

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class TrainingSummary(metaclass=Singleton):

    def __init__(self):
        self.id = str(uuid4())
        self.thumbnail_dir = os.path.normpath('../../data/GTSRB/processed/thumbnail')
        self.model_dir = os.path.normpath('../../models/lenet/')

    def load(self, model_name):
        current_model_dir = os.path.normpath(os.path.join('../../models/lenet/', model_name))
        errors_dir = os.path.normpath(os.path.join(current_model_dir, 'errors'))

        self.errors = list()
        with open(os.path.join(errors_dir, 'annotations.pkl'), 'rb') as file_pickle:
            while True:
                try:
                    self.errors.append(pickle.load(file_pickle))
                except EOFError:
                    break
        self.stats = self.errors.pop()

        self.classes = list()
        pathlist = sorted(Path(self.thumbnail_dir).glob('**/*.jpg'))
        for path in pathlist:
            path_in_str = str(path)
            components = os.path.split(path_in_str)
            filename = components[1]
            self.classes.append(filename)
        self.classes.sort()

    def get_classes(self):
        return self.classes
    def get_errors(self):
        return self.errors
    def get_stats(self):
        return self.stats
    def get_thumbnail_dir(self):
        return self.thumbnail_dir
    def get_model_dir(self):
        return self.model_dir
    def get_model_diagram(self):
        return 'model.png'
    def get_history_diagram(self):
        return 'history.png'
    def get_id(self):
        return self.id

is_closing = False

def signal_handler(signum, frame):
    global is_closing
    logging.info('exiting...')
    is_closing = True

def try_exit(): 
    global is_closing
    if is_closing:
        # clean up here
        tornado.ioloop.IOLoop.instance().stop()
        print('Terminated')

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        print('MainHandler::get()', self.request.uri, self.request.host, self.request.full_url())
        cgi_prefix = ''
        if self.request.host.startswith('cgibackend'):
            cgi_prefix = 'cgi/'
        loader = tornado.template.Loader("./server/templates")
        html = loader.load("index.html").generate(cgi_prefix=cgi_prefix)
        self.write(html)

class GTSRBHandler(tornado.web.RequestHandler):
    def get(self, path):
        components = os.path.split(path)
        model_name = components[1]
        TrainingSummary().load(model_name)
        
        loader = tornado.template.Loader("./templates")
        html = loader.load("gtsrb.html").generate(
            model_name = model_name,
            classes = TrainingSummary().get_classes(),
            errors = TrainingSummary().get_errors(),
            stats = TrainingSummary().get_stats(),
            model_diagram = TrainingSummary().get_model_diagram(),
            history_diagram = TrainingSummary().get_history_diagram(),
        )
        self.write(html)

class ImageHandler(tornado.web.RequestHandler):
    def get(self, path):
        prefix = path[:path.find('\\')]
        prefix = path[:path.find('/')]
        trailing = path[path.find('\\')+1:]
        trailing = path[path.find('/')+1:]
        components = os.path.split(trailing)
        directory = components[0]
        filename = components[1]
        if prefix=='thumbnail':
            directory = TrainingSummary().get_thumbnail_dir()
        elif prefix=='model':
            directory = os.path.normpath(os.path.join(TrainingSummary().get_model_dir(), directory))

        inpath = os.path.normpath(os.path.join(directory, filename))
        try:
            with open(inpath, 'rb') as imgf:
                modified = datetime.datetime.fromtimestamp(os.path.getmtime(inpath))
                self.set_header('Last-Modified', modified)
                self.set_header('Expires', datetime.datetime.utcnow() + \
                    datetime.timedelta(days=1))
                self.set_header("Cache-Control", 'max-age=' + str(86400*1))
                header = 'image/jpeg'
                self.add_header('Content-Type', header) 
                self.write(imgf.read())
                return
        except Exception:
            print('I/O Exception:', inpath)
            pass

        self.clear()
        self.set_status(404)
        self.finish('Image not found')

class StaticHandler(tornado.web.RequestHandler):
    def get(self, path):
        path = os.path.normpath(os.path.join('./server/static', path))
        print('StaticHandler', path)
        try:
            with open(path, 'rb') as f:
                """modified = datetime.datetime.fromtimestamp(os.path.getmtime(path))
                self.set_header('Last-Modified', modified)
                self.set_header('Expires', datetime.datetime.utcnow() + \
                    datetime.timedelta(days=1))
                self.set_header("Cache-Control", 'max-age=' + str(86400*1))"""
                extension = os.path.splitext(os.path.split(path)[1])[1]
                if extension=='.js':
                    type = 'application/javascript'
                self.add_header('Content-Type', type)
                self.write(f.read())
                return
        except Exception:
            print('I/O Exception:', path)
            pass

        self.clear()
        self.set_status(404)
        self.finish('File not found')

class PredictHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header('Content-Type', 'application/json')

    def __format_image(self, bindata):
        """
        - Convert portable text media to pixel array
        - Object media is JSON compliant
        - Object media has one of the members:
          'url' - URL to an online source
          'content' - BASE64 encoded JPEG data
        - Formatted pixel array for CNN is in RGB channel order
        - Formatted pixel array for CNN has dtype=np.float
        - Formatted pixel array for CNN has pixel values normalized to range [0, 1]
        """
        img = None
        img = cv2.imdecode(np.frombuffer(bindata, np.uint8), 1)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img.astype(np.float32)) / 255.
        return img

    def get(self):
        self.set_status(405)
        self.finish('HTTP GET is not supported; use POST instead. Please compose requests following the guide of API reference at http://project.arobot.info/redmine/documents/19')

    def post(self):
        #self.get_argument('username')
        t_now = time.time() * 1000
        info('POST request from: ' + self.request.remote_ip)

        try:
            self.out_queue
        except AttributeError:
            self.out_queue = Queue()

        requests = None
        img = None

        print()
        print()
        print(self.request.headers)
        print()

        if len(self.request.body_arguments):
            # multipart/form-data
            print('multipart/form-data')
            for name in self.request.body_arguments:
                part = self.request.body_arguments[name][0]
                print(name, part[0:32])
                print()
                if name=='requests':
                    requests = part
                elif name=='media':
                    if len(part) > 2*1024*1024:
                        self.set_status(413)
                        self.finish('Content of media too large')
                        return
                    img = self.__format_image(part)
                    if img is None:
                        self.set_status(415)
                        self.finish("Unable to load media")
                        return

        elif self.request.body:
            # Base64-encode media and store in JSON
            requests = self.request.body

        try:
            requests = requests.decode('utf-8')
            print(requests)
        except ValueError:
            self.set_status(400)
            self.finish('Unable to decode as UTF-8')
            self.log_exception()
            return

        print()
        print()

        try:
            requests = json.loads(requests)
        except ValueError:
            self.set_status(400)
            self.finish('Unable to parse JSON')
            self.log_exception()
            return

        if requests is not None and img is not None:
            try:
                if 'agent' in requests and 'debug' in requests['agent'] and requests['agent']['debug']:
                    original_request = copy.deepcopy(requests)
                    for request in original_request['requests']:
                        if 'media' in request and 'content' in request['media']:
                            request['media']['content_length'] = len(request['media']['content'])
                            request['media'].pop('content', None)
                        else:
                            request['media'] = 'DEBUG_EMPTY'
                    json_str_debug = json.dumps(original_request, cls=NumpyEncoder)
                    debug('Detection request ' + json_str_debug)

                if 'timing' in requests:
                    requests['timing']['server_rcv'] = time.time() * 1000

                """
                - One or multiple predict request could be included in single HTTP POST
                - Each predict request has only one 'media', which contains an image
                - Each predict request requests one or multiple services
                - For each service, 'type' and 'model' must be specified
                - For each service type, options are type dependent
                """
                if 'requests' not in requests:
                    self.set_status(400)
                    self.finish("Requests must be enclosed in 'requests' attribute as a list of request")
                    return
                for request in requests['requests']:
                    if 'services' not in request:
                        self.set_status(400)
                        self.finish("Services must be enclosed in 'services' attribute for request " + request['requestId'])
                        return

                    for service in request['services']:
                        #print(service)
                        service_timing = {}

                        if 'type' not in service:
                            self.set_status(400)
                            self.finish("'type' attribute of a service must be specified for request " + request['requestId'])
                            return

                        # Decode BASE64 encoded JPEG image in JSON
                        t_ = time.time()
                        if img is None:
                            if 'media' not in request:
                                self.set_status(400)
                                self.finish('Media is empty for request ' + request['requestId'])
                                return
                            if 'content' in request['media']:
                                # Hard-cap length of media/contant at 2MB to save server resource
                                if len(request['media']['content']) > 2*1024*1024:
                                    self.set_status(413)
                                    self.finish('Content of media too large for request ' + request['requestId'])
                                    return
                                img = self.__format_image(base64.b64decode(request['media']['content'].encode()))
                            elif 'url' in request['media']:
                                raise(NotImplementedError)
                                pass
                        if img is None:
                            self.set_status(415)
                            self.finish("Unable to load media for request " + request['requestId'])
                            debug('Unable to load media {}'.format(request['media']))
                            return
                        service_timing['decode_img'] = (time.time() - t_) * 1000
                        
                        if service['type']=='_void':
                            # For validating and benchmarking network connection
                            service['results'] = {
                                'timing': service_timing,
                            }
                        elif service['type']=='face':
                            face_app = FaceApplications()
                            params = {
                                'service': service,
                                'output_holder': self.out_queue,
                                'agent': requests['agent']
                            }
                            if face_app.detect(img, params=params):
                                # Call get() to block and wait for detection result
                                t_ = time.time()
                                output = self.out_queue.get() # This is blocked until something is put in the queue
                                #print('.get() latency', (time.time() - t_) * 1000)
                            else:
                                # Detection threadings are busy
                                self.set_status(429)
                                self.finish('Server is busy')
                                return

                            if output is None: # Detection thread skips this frame
                                self.set_status(429)
                                self.finish('Too many requests')
                                return

                            service['results'] = output['predictions']
                            if 'options' in service: service.pop('options', None)
                            self.out_queue.task_done()
                            #print('service', service)
                        elif service['type']=='face_':
                            # This is for testing before 201803
                            classifier = FaceClassifier()
                            classifier.init()
                            rects, predictions, timing, fdetect_result = classifier.detect(request['media'])
                            i = 0
                            for rect in rects:
                                print(rect, predictions[i])
                                i += 1
                            service['results'] = {
                                'rects': rects,
                                'predictions': predictions,
                                'mtcnn': fdetect_result['mtcnn'],
                                'mtcnn_5p': fdetect_result['mtcnn_5p'],
                                'emotions': fdetect_result['emotions'],
                                'timing': timing,
                            }
                    request.pop('media', None)

            except:
                self.set_status(500)
                self.log_exception()
                return
    
            #print('json decoded', request)

        # Set up response dictionary.
        #self.response = request
        #print(self.response)
        if 'timing' in requests:
            requests['timing']['server_sent'] = time.time()*1000

        requests['responses'] = requests['requests']
        requests.pop('requests', None)

        json_str = json.dumps(requests, cls=NumpyEncoder)

        info('Response, latency: {}'.format(time.time()*1000-t_now))

        self.write(json_str)
        self.finish()

    def log_exception(self):
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error('\n'.join(['Exception: {}'.format(threading.current_thread())] + list(traceback.format_tb(exc_traceback, limit=32)) + [exc_type.__name__+': '+str(exc_value),]))

def make_app():
    return tornado.web.Application([
        (r"/gtsrb/(.*)", GTSRBHandler),
        (r"/img/(.*)", ImageHandler),
        (r"/static/(.*)", StaticHandler),
        (r"/predict", PredictHandler),
        (r"/", MainHandler),
    ])

def server_start(args, port=9000):
    info('Serving tornado server on port ' + str(port))
    FaceApplications() # Initialize face applications singleton
    app = make_app()
    app.listen(port)
    tornado.ioloop.PeriodicCallback(try_exit, 100).start()
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    server_start(None)


