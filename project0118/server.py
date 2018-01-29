import tornado.ioloop
import tornado.web
import tornado.template

import time
from uuid import uuid4
import os.path
from pathlib import Path
import pickle

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

    def load(self, model_name):
        self.model_dir = os.path.normpath(os.path.join('../../models/lenet/', model_name))
        self.errors_dir = os.path.normpath(os.path.join(self.model_dir, 'errors'))

        self.errors = list()
        with open(os.path.join(self.errors_dir, 'annotations.pkl'), 'rb') as file_pickle:
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
    def get_errors_dir(self):
        return self.errors_dir
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
        loader = tornado.template.Loader("./templates")
        html = loader.load("index.html").generate(
            classes = TrainingSummary().get_classes(),
            errors = TrainingSummary().get_errors(),
            stats = TrainingSummary().get_stats(),
            model_diagram = TrainingSummary().get_model_diagram(),
            history_diagram = TrainingSummary().get_history_diagram(),
        )
        self.write(html)

class GTSRBHandler(tornado.web.RequestHandler):
    def get(self, path):
        components = os.path.split(path)
        model_name = components[1]
        TrainingSummary().load(model_name)
        
        loader = tornado.template.Loader("./templates")
        html = loader.load("gtsrb.html").generate(
            classes = TrainingSummary().get_classes(),
            errors = TrainingSummary().get_errors(),
            stats = TrainingSummary().get_stats(),
            model_diagram = TrainingSummary().get_model_diagram(),
            history_diagram = TrainingSummary().get_history_diagram(),
        )
        self.write(html)

class ImageHandler(tornado.web.RequestHandler):
    def get(self, path):
        components = os.path.split(path)
        if components[0]=='thumbnail':
            directory = TrainingSummary().get_thumbnail_dir()
        elif components[0]=='model':
            directory = TrainingSummary().get_model_dir()
        elif components[0]=='errors':
            directory = TrainingSummary().get_errors_dir()

        filename = components[1]
        inpath = os.path.normpath(os.path.join(directory, filename))
        try:
            with open(inpath, 'rb') as imgf:
                header = 'image/jpeg'
                self.add_header('Content-Type', header) 
                self.write(imgf.read())
                return
        except Exception:
            print('I/O Exception:', inpath)
            pass

        self.clear()
        self.set_status(404)
        self.finish("Image not found")

def make_app():
    return tornado.web.Application([
        (r"/gtsrb/(.*)", GTSRBHandler),
        (r"/img/(.*)", ImageHandler),
    ])

if __name__ == "__main__":
    port = 8888
    print('Serving tornado server on port', port)
    app = make_app()
    app.listen(port)
    tornado.ioloop.PeriodicCallback(try_exit, 100).start()
    tornado.ioloop.IOLoop.current().start()


