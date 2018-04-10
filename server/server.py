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

from facer.predict import FaceClassifier
from facer.face_app import FaceApplications

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
        print('StaticHandler', path)
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

    def __format_image(self, media):
        """
        - Convert portable text media to pixel array
        - Object media is JSON compliant
        - Object media has one of the members:
          'url' - URL to an online source
          'content' - Base 64 encoded JPEG data
        - Formatted pixel array for CNN is in RGB channel order
        - Formatted pixel array for CNN has dtype=np.float
        - Formatted pixel array for CNN has pixel values normalized to range [0, 1]
        """
        img = None
        if 'content' in media:
            bindata = base64.b64decode(media['content'].encode())
            img = cv2.imdecode(np.frombuffer(bindata, np.uint8), 1)
        elif 'url' in media:
            raise(NotImplementedError)
            pass
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float)) / 255.
        return img

    def get(self):
        self.set_status(405)
        self.finish('HTTP GET is not supported; use POST instead. Please compose requests following the guide of API reference at http://project.arobot.info/redmine/documents/19')

    def post(self):
        #self.get_argument('username')
        if self.request.body:
            try:
                postdata = self.request.body.decode('utf-8')
            except ValueError:
                self.set_status(400)
                self.finish('Unable to decode as UTF-8')
                return

            try:
                json_data = json.loads(postdata)
            except ValueError:
                self.set_status(400)
                self.finish('Unable to parse JSON')
                return

            if 'timing' in json_data:
                json_data['timing']['server_rcv'] = time.time() * 1000

            print()
            """
            - One or multiple predict request could be included in single HTTP POST
            - Each predict request has only one 'media', which contains an image
            - Each predict request requests one or multiple services
            - For each service, 'type' and 'model' must be specified
            - For each service type, options are type dependent
            """
            if 'requests' not in json_data:
                self.set_status(400)
                self.finish("Requests must be enclosed in 'requests' attribute as a list of request")
                return
            for request in json_data['requests']:
                if 'services' not in request:
                    self.set_status(400)
                    self.finish("Services must be enclosed in 'services' attribute for request " + request['requestId'])
                    return

                img = None

                for service in request['services']:
                    print(service)
                    service_timing = {}

                    if 'media' not in request:
                        self.set_status(400)
                        self.finish('Media is empty for request ' + request['requestId'])
                        return
                    if 'content' in request['media']:
                        # Hard-cap length of media/contant at 2MB to save server resource
                        if len(request['media']['content']) > 2*1024*1024:
                            self.set_status(400)
                            self.finish('Content of media too large for request ' + request['requestId'])
                            return

                    if 'type' not in service:
                        self.set_status(400)
                        self.finish("'type' attribute of a service must be specified for request " + request['requestId'])
                        return

                    t_ = time.time()
                    if img is None: img = self.__format_image(request['media'])
                    if img is None:
                        self.set_status(400)
                        self.finish("Unable to load media for request " + request['requestId'])
                        return
                        self.send_error(400, message="Invalid 'media'") # Bad Request
                        self.finish()
                    service_timing['decode_img'] = (time.time() - t_) * 1000
                    
                    if service['type']=='_void':
                        # For validating and benchmarking network connection
                        service['results'] = {
                            'timing': service_timing,
                        }
                    elif service['type']=='face':
                        face_app = FaceApplications()
                        predictions = face_app.detect(img, params={
                            'service': service,
                        })
                        service['results'] = predictions
                        if 'options' in service: service.pop('options', None)
                    elif service['type']=='face_':
                        # This is for testing before 201803
                        classifier = FaceClassifier()
                        classifier.init()
                        rects, predictions, timing, fdetect_result = classifier.detect(request['media'])
                        print(timing)
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
    
            #print('json decoded', json_data)

        # Set up response dictionary.
        #self.response = json_data
        #print(self.response)
        if 'timing' in json_data:
            json_data['timing']['server_sent'] = time.time()*1000

        json_data['responses'] = json_data['requests']
        json_data.pop('requests', None)

        self.write(json.dumps(json_data))
        self.finish()

def make_app():
    return tornado.web.Application([
        (r"/gtsrb/(.*)", GTSRBHandler),
        (r"/img/(.*)", ImageHandler),
        (r"/static/(.*)", StaticHandler),
        (r"/predict", PredictHandler),
        (r"/", MainHandler),
    ])

def server_start(args, port=9000):
    print('Serving tornado server on port', port)
    app = make_app()
    app.listen(port)
    tornado.ioloop.PeriodicCallback(try_exit, 100).start()
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    server_start(None)


