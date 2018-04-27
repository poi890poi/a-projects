from mtcnn import detect_face as FaceDetector
from facer.emotion import EmotionClassifier
from shared.utilities import ImageUtilities as imutil
from shared.utilities import DirectoryWalker as dirwalker
from shared.utilities import NumpyEncoder
from shared.alogger import *

from facenet import facenet
import sklearn.metrics, sklearn.preprocessing
import scipy.spatial, scipy.cluster
from sklearn.utils import shuffle

import sys
import time
import numpy as np
import cv2
import tensorflow as tf
import os.path
import pathlib
import math
import uuid
import json
import random
import copy
import base64

from queue import Queue, Empty
import threading
import traceback

FACE_EMBEDDING_THRESHOLD_HIGH = 0.5375 # High precision 99%
FACE_EMBEDDING_THRESHOLD_RECALL = 0.584375 # High recall 96%. This is used for querying to optimize recall
FACE_EMBEDDING_THRESHOLD_LOW = 0.378125 # Very high precision 99.866777%. This is used for finding existing identity during registering new faces

# The number of face samples required for registration of an identity
# (additional FACE_EMBEDDING_SAMPLE_SIZE will be added later for increase accuracy and keeps record up to date)
FACE_EMBEDDING_SAMPLE_SIZE = 6

FACE_RECOGNITION_CONCURRENT = 1
FACE_RECOGNITION_REMEMBER = 32
FACE_RECOGNITION_DIR = '../models/face_registered'
INTERVAL_TASK_EXPIRATION = 500. # Task expiration time before being processed by a thread
INTERVAL_FACE_REGISTER = 2000.
INTERVAL_FACE_SAVE = 300. * 1000
RUN_MODE_DEBUG = True

# Lazy mechanism use results of previous detection if time difference
# of consecutive frame is within the constants defined here (in seconds)
LAZY_DETECTION = 125.
LAZY_RECOGNITION = 2000.

"""
- Array plane is (custom) defined as coordinate system of y (row) before x (col),
  optimizing for array and bytes operations
- Image plane is (custom) defined as coordinate system of x (col) before y (row),
  optimizing for image operations, specifically OpenCV compatibility
"""

class Rectangle:
    """
    Rectangle is (custom) defined with upper-left point (y, x) and length of 2 sides (h, w)
    """
    def __init__(self, y, x, h, w):
        self.rectangle = np.array([y, x, h, w], dtype=np.float)

    def to_extent(self):
        return Extent(self.rectangle[0], self.rectangle[1], self.rectangle[0]+self.rectangle[2], self.rectangle[1]+self.rectangle[3])

    def eval(self):
        return self.rectangle

class Extent:
    """
    Extent is (custom) defined with upper-left point (y1, x1) and lower-right point (y2, x2)
    """
    def __init__(self, y1, x1, y2, x2):
        self.extent = np.array([y1, x1, y2, x2], dtype=np.float)

    def to_rectangle(self):
        return Rectangle(self.extent[0], self.extent[1], self.extent[2]-self.extent[0], self.extent[3]-self.extent[1])

    def eval(self):
        return self.extent

class ThreadTask():
    """
    A task is a container to hold input parameters and output results for detection.
    It's pushed into the synchronized queue to be staged for detection.
    A task only holds variables and does not do intensive computing.
    """
    def __init__(self, img=None, params=None):
        self.img = img
        self.params = params
        self.t_queued = time.time() * 1000
        self.t_expiration = self.t_queued + INTERVAL_TASK_EXPIRATION
        
        # For sending embeddings from detection thread to embedding thread
        self.embeddings = None

        # For sending data to be saved to FileWritingThread
        self.write = None

class ThreadBase(threading.Thread):
    """
    A core is a thread class that manages model and does actual detection.
    Multiple instances of a core class allocate multiple copies of Tensorflow session, if it's used.
    Cores are managed by 'applications' class objects.
    """
    def __init__(self, in_queue):
        print(threading.current_thread(), 'ThreadBase::__init__()')
        threading.Thread.__init__(self)
        self.is_standby = False
        self.is_crashed = False
        self.in_queue = in_queue
        self._init_derived()

    def _on_crash(self):
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error('\n'.join(['Thread Exception: {}'.format(threading.current_thread())] + list(traceback.format_tb(exc_traceback, limit=32)) + [exc_type.__name__+': '+str(exc_value),]))
        self.is_crashed = True

    def _init_derived(self):
        raise NotImplementedError('VisionCore::__init_derived() must be implemented in derived class')

    def run(self):
        raise NotImplementedError('VisionCore::run() must be implemented in derived class')

        # This is a template. Every thread must implement run() and modify this based on their jobs.
        while True:
            if self.is_crashed:
                # The thread is broken; do nothing...
                # or restart the thread in main thread, if its job is so important.
                time.sleep(1)
                continue
    
            if not self.is_standby:
                # Do initialization here...
                self.is_standby = True
                continue

            try:
                t_now = time.time() * 1000 # Timing is essential for threading. It's a good practice to keep track of processing time for each thread.
                task = self.in_queue.get() # queue::get() is blocking by default

                # Do its job...

                self.in_queue.task_done() # Must be called or the queue will be fulled eventually
            except:
                self._on_crash()


class FaceEmbeddingAgent():
    """
    FaceEmbeddingAgent is an agent-specific object that stores required data for face embedding registering,
    a process that samples face images, checks if the face is already in database of known faces, add the
    new faces to database, and create KD tree of the faces to be searched.
    """

    def __init__(self, agent_id, thread):
        self.agent_id = agent_id
        
        # Prepare directory
        self.safe_id = base64.urlsafe_b64encode(self.agent_id.encode()).decode()
        if not os.path.isdir(FACE_RECOGNITION_DIR):
            pathlib.Path(FACE_RECOGNITION_DIR).mkdir(parents=True, exist_ok=True)
        self.dir_embeddings = FACE_RECOGNITION_DIR + '/' + self.safe_id
        if not os.path.isdir(self.dir_embeddings):
            pathlib.Path(self.dir_embeddings).mkdir(parents=True, exist_ok=True)
        self.fn_embeddings = self.dir_embeddings + '/embeddings.json'

        self.thread = thread
        self.t_last_frame = 0
        self.is_registering = False

        self.t_update_face_tree = 0
        self.t_save_face_tree = 0
        self.t_save_face_images = 0

        self.face_embeddings_candidate = {
            'embeddings': [],
            'rectangles': [],
            'face_images': [],
            'tree': None,
        }

        self.face_names_cluster = []
        self.face_embeddings_cluster = []
        self.face_images_cluster = []
        self.face_tree_cluster = None
        self.names_to_index = {}
        self.is_tree_dirty = False
        self.is_tree_outsync = False

    def append_cluster(self, index, embedding, image):
        # Update an existing cluster with latest face embeddings that match the cluster
        #print('append_cluster', index, len(embedding), len(image))
        self.face_embeddings_cluster[index].append(embedding)
        if index < len(self.face_images_cluster):
            self.face_images_cluster[index].append((np.array(image)*255).astype(dtype=np.uint8))

        # Mix initial samples and appending samples, while keeping at least FACE_EMBEDDING_SAMPLE_SIZE of initial samples
        if len(self.face_embeddings_cluster[index]) > FACE_EMBEDDING_SAMPLE_SIZE * 3:
            if index < len(self.face_images_cluster) and len(self.face_images_cluster[index])==len(self.face_embeddings_cluster[index]):
                embeddings_shuffle_trim = self.face_embeddings_cluster[index][FACE_EMBEDDING_SAMPLE_SIZE::]
                images_shuffle_trim = self.face_images_cluster[index][FACE_EMBEDDING_SAMPLE_SIZE::]
                embeddings_shuffle_trim, images_shuffle_trim = shuffle(embeddings_shuffle_trim, images_shuffle_trim)
                self.face_embeddings_cluster[index] = self.face_embeddings_cluster[index][0:FACE_EMBEDDING_SAMPLE_SIZE] + embeddings_shuffle_trim[0:FACE_EMBEDDING_SAMPLE_SIZE]
                self.face_images_cluster[index] = self.face_images_cluster[index][0:FACE_EMBEDDING_SAMPLE_SIZE] + images_shuffle_trim[0:FACE_EMBEDDING_SAMPLE_SIZE]
            else:
                embeddings_shuffle_trim = self.face_embeddings_cluster[index][FACE_EMBEDDING_SAMPLE_SIZE::]
                embeddings_shuffle_trim = shuffle(embeddings_shuffle_trim)
                self.face_embeddings_cluster[index] = self.face_embeddings_cluster[index][0:FACE_EMBEDDING_SAMPLE_SIZE] + embeddings_shuffle_trim[0:FACE_EMBEDDING_SAMPLE_SIZE]

        self.is_tree_dirty = True

    def register_new_cluster(self, cluster, images):
        # A new face is found
        info('register_new_cluster, shape: {}'.format(np.array(cluster).shape))
        name = str(uuid.uuid4()) # Assign a random new name
        self.names_to_index[name] = len(self.face_embeddings_cluster)
        self.face_embeddings_cluster.append(cluster)
        self.face_names_cluster.append([name, len(cluster)])
        if RUN_MODE_DEBUG:
            self.face_images_cluster.append((np.array(images)*255).astype(dtype=np.uint8).tolist())
        
        while len(self.face_embeddings_cluster) > FACE_RECOGNITION_REMEMBER: # Maximum of faces to remember
            name = self.face_names_cluster.pop(0)
            self.face_embeddings_cluster.pop(0)
            self.face_names_cluster.pop(name)
            if RUN_MODE_DEBUG:
                self.face_images_cluster.pop(0)

        self.is_tree_dirty = True

    def check_update_tree(self):
        #print('check_update_tree', self.is_tree_dirty)
        try:
            t_now = time.time() * 1000
            if self.is_tree_dirty:

                print('check_update_tree', len(self.face_embeddings_cluster), len(self.face_images_cluster))
                # Flatten the list of lists with varying length
                emb_flatten = []
                #print('names', self.face_names_cluster)
                for index, cluster in enumerate(self.face_embeddings_cluster):
                    # Limit length of each cluster
                    
                    """ # The trimming and shuffling was done in append_cluster() to prevent images from accumulating in memory
                    # Shuffle before trimming to mix old and new samples
                    if index < len(self.face_images_cluster) and len(self.face_embeddings_cluster[index])==len(self.face_images_cluster[index]):
                        #self.face_embeddings_cluster[index], self.face_images_cluster[index] = shuffle(self.face_embeddings_cluster[index], self.face_images_cluster[index])
                        while len(self.face_embeddings_cluster[index]) > FACE_EMBEDDING_SAMPLE_SIZE * 2: # Maximum of samples for a face
                            self.face_embeddings_cluster[index].pop(0)
                            self.face_images_cluster[index].pop(0)
                    else:
                        self.face_embeddings_cluster[index] = shuffle(self.face_embeddings_cluster[index])
                        while len(self.face_embeddings_cluster[index]) > FACE_EMBEDDING_SAMPLE_SIZE * 2: # Maximum of samples for a face
                            self.face_embeddings_cluster[index].pop(0)
                    """

                    self.face_names_cluster[index][1] = len(self.face_embeddings_cluster[index])

                for index, cluster in enumerate(self.face_embeddings_cluster):
                    # Flatten cluster so it can be fed to scipy.spatial.KDTree()
                    for emb in cluster:
                        emb_flatten.append(emb)

                debug('Tree updated, identities: '.format(self.face_names_cluster))
                self.face_tree_cluster = scipy.spatial.KDTree(np.array(emb_flatten).reshape(-1, 128))
                FaceApplications().tree_updated(self.agent_id, self.face_tree_cluster, self.face_names_cluster)
                self.is_tree_outsync = True
                if self.t_save_face_tree==0: self.t_save_face_tree = t_now + INTERVAL_FACE_SAVE

            self.is_tree_dirty = False

            if self.is_tree_outsync:
                debug('Check saving embeddings to file... {}'.format(t_now-self.t_save_face_tree))
                if self.t_save_face_tree > 0 and t_now > self.t_save_face_tree:
                    print()
                    print()
                    print(threading.current_thread(), 'Saving embeddings to file...', len(self.face_images_cluster))
                    debug('Saving embeddings to file... {}'.format(len(self.face_images_cluster)))
                    # Save registered faces as files
                    # This is for debugging and has significant impact on performance
                    for index, images in enumerate(self.face_images_cluster):
                        name, count = self.face_names_cluster[index]
                        dir = self.dir_embeddings + '/' + name
                        if not os.path.isdir(dir):
                            os.makedirs(dir)
                        debug('images, name: {}, images: {}'.format(name, len(images)))
                        for f_seq, f_img in enumerate(images):
                            #img = (np.array(f_img)*255).astype(dtype=np.uint8)
                            f_img = np.array(f_img, dtype=np.uint8)
                            img = cv2.cvtColor(f_img, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(dir+'/'+str(f_seq).zfill(4)+'.jpg', img)

                    json_obj = {
                        'embeddings': self.face_embeddings_cluster,
                        'names': self.face_names_cluster,
                        #'images': self.face_images_cluster,
                        'names_to_index': self.names_to_index,
                    }
                    if FaceApplications().file_write(self.fn_embeddings, json.dumps(json_obj, cls=NumpyEncoder)):
                        self.is_tree_outsync = False

                    self.t_save_face_tree = 0
        except:
            self.thread._on_crash()
            #exc_type, exc_value, exc_traceback = sys.exc_info()
            #error('\n'.join(['Thread Exception: {}'.format(threading.current_thread())] + list(traceback.format_tb(exc_traceback, limit=32)) + [exc_type.__name__+': '+str(exc_value),]))

    def restore_embeddings(self):
        # Load face embeddings from disk and update search tree
        print(threading.current_thread(), 'Restoring embeddings from file...')
        try:
            with open(self.fn_embeddings, 'r') as fr:
                json_obj = json.loads(fr.read())
            self.face_embeddings_cluster = json_obj['embeddings']
            self.face_names_cluster = json_obj['names']
            if 'images' in json_obj: self.face_images_cluster = json_obj['images']
            if not RUN_MODE_DEBUG:
                self.face_images_cluster = []
            self.names_to_index = json_obj['names_to_index']
            self.is_tree_dirty = True
            self.check_update_tree()
        except FileNotFoundError:
            pass
        print(threading.current_thread(), 'Embeddings restored.')

class FileWritingThread(ThreadBase):
    def _init_derived(self):
        pass

    def run(self):
        while True:
            if self.is_crashed:
                # The thread is broken; do nothing...
                # or restart the thread in main thread, if its job is so important.
                time.sleep(1)
                continue
    
            if not self.is_standby:
                # Do initialization here...
                self.is_standby = True
                continue

            try:
                t_now = time.time() * 1000 # Timing is essential for threading. It's a good practice to keep track of processing time for each thread.
                task = self.in_queue.get() # queue::get() is blocking by default

                if task.write is not None:
                    filepath = task.write['filepath']
                    content = task.write['content']
                    with open(filepath, 'w') as fw:
                        fw.write(content)
                        info('{} bytes of data written to file {} succesfully'.format(len(content), filepath))

                self.in_queue.task_done() # Must be called or the queue will be fulled eventually
            except:
                self._on_crash()

class FaceEmbeddingThread(ThreadBase):
    """
    FaceEmbeddingThread:
    1. Aggregate face embeddings
    2. Group them with intersection of union (IOU) as clusters
    3. Search each cluster in database of known faces
    4. If not found, register the cluster to database of known faces
    """

    def _init_derived(self):
        self.agents = {}
        #print(threading.current_thread(), 'FaceEmbeddingThread::_init_derived()')

    def run(self):
        while True:
            if self.is_crashed:
                # The thread is broken; do nothing
                time.sleep(1)
                continue

            if not self.is_standby:
                # Do initialization here...
                self.is_standby = True
                continue
            
            try:
                t_now = time.time() * 1000

                # Get new face embeddings and update KDTree
                task = self.in_queue.get() # queue::get() is blocking by default
                #print('get embedding', t_now-task.t_expiration)

                if t_now > task.t_expiration:
                    warning('Skip outdated embedding, delay: {}, embeddings: {}'.format(t_now-task.t_expiration, (task.embeddings is not None)))
                    pass
                    
                elif task.embeddings:
                    agent_id = task.embeddings['agent']['agentId']
                    if agent_id not in self.agents:
                        self.agents[agent_id] = FaceEmbeddingAgent(agent_id, self)
                        self.agents[agent_id].restore_embeddings()
                    agent = self.agents[agent_id]

                    embeddings = sklearn.preprocessing.normalize(task.embeddings['recognition']['embeddings'])
                    names = task.embeddings['names']
                    face_images = task.embeddings['recognition']['images']
                    rectangles = task.embeddings['recognition']['rectangles']
                    is_registering = False
                    if task.embeddings['recognition']['mode']=='register': is_registering = True
                    print('is registering', is_registering, task.embeddings['recognition']['mode'])

                    #print('queried names', names)
                    #print('get embedding', len(embeddings), len(face_images), len(rectangles))

                    candidate = agent.face_embeddings_candidate

                    if is_registering:
                        if not agent.is_registering:
                            info('Engage registering... {}'.format(agent.safe_id))
                            # Reset candidates when engaging a new session of registering, to prevent mixing faces of different identities
                            candidate['embeddings'] = []
                            candidate['face_images'] = []
                            candidate['rectangles'] = []
                    agent.is_registering = is_registering

                    for emb_i, emb in enumerate(embeddings):
                        if names[emb_i]:
                            # The face is found in registered faces; append to the existing cluster
                            index = agent.names_to_index[names[emb_i]]
                            #print('registered faces', names[emb_i], len(agent.face_names_cluster), len(self.face_embeddings_cluster), len(self.face_images_cluster))
                            agent.append_cluster(index, emb, face_images[emb_i])
                        else:
                            # The face is unknown
                            if is_registering:
                                candidate['embeddings'].append(emb)
                                candidate['face_images'].append(face_images[emb_i])
                                candidate['rectangles'].append(rectangles[emb_i])

                    if not is_registering:
                        # Update appended samples of existing identity to keep the record up to date
                        if agent.t_update_face_tree==0:
                            agent.t_update_face_tree = t_now + INTERVAL_FACE_REGISTER * 4 # Querying mode updates tree less frequently
                        elif t_now > agent.t_update_face_tree:
                            agent.check_update_tree()
                            agent.t_update_face_tree = 0

                    else:
                        # Each agent samples a limit number of faces for candidates to be registered
                        while len(candidate['embeddings']) > FACE_EMBEDDING_SAMPLE_SIZE * 2:
                            candidate['embeddings'].pop(0)
                            candidate['face_images'].pop(0)
                            candidate['rectangles'].pop(0)

                        #print(threading.current_thread(), 'CHECK UPDATE FACE TREE', t_now-self.t_update_face_tree)
                        # Update KD Tree only every X seconds
                        debug('Check registering faces, time_delta: {}'.format(t_now-agent.t_update_face_tree))
                        if agent.t_update_face_tree==0:
                            agent.t_update_face_tree = t_now + INTERVAL_FACE_REGISTER
                            debug('Signal faces registering')
                        elif t_now > agent.t_update_face_tree:
                            debug('Check registering new identity... {}'.format(agent.safe_id))
                            # Group face embeddings into clusters

                            # 20180419 Lee noted: query_ball_tree() of scipy returns a lot of duplicated clusters and the result is also suboptimal
                            # Use IOU instead to group candidates
                            #self.face_tree_register = scipy.spatial.KDTree(register['embeddings'])
                            #clusters = self.face_tree_register.query_ball_tree(self.face_tree_register, FACE_EMBEDDING_THRESHOLD_LOW)
                            clusters = imutil.group_rectangles_miniou(candidate['rectangles'], threshold=0.3)
                            if clusters is not None:
                                #print(clusters)
                                # Sorting does nothing right now; all clusters are processed, independently
                                #clusters.sort(key=len, reverse=True)
                                #print('sorted', clusters)
                                for ci, new_cluster in enumerate(clusters):
                                    if len(new_cluster) >= FACE_EMBEDDING_SAMPLE_SIZE:
                                        len_ = len(new_cluster)

                                        # Discard clusters with faces at different locations; this has no use for IOU clustering
                                        """ # This is required only for clusters from query_ball_tree() of scipy, which is no longer used due to slow performance and suboptimal results
                                        discard = False
                                        for fi1 in new_cluster:
                                            for fi2 in new_cluster:
                                                if fi1!=fi2:
                                                    r1 = candidate['rectangles'][fi1]
                                                    r2 = candidate['rectangles'][fi2]
                                                    iou = imutil.calc_iou(r1, r2)
                                                    if iou < 0.5:
                                                        #print('small IOU, discard', iou)
                                                        discard = True
                                                        break
                                            if discard: break
                                        if discard: continue
                                        """
                                        print('the cluster is valid', ci, len(new_cluster), new_cluster)

                                        # Convert index to 128-bytes embedding
                                        face_images = []
                                        for i, member in enumerate(new_cluster):
                                            new_cluster[i] = candidate['embeddings'][member]
                                            face_images.append(candidate['face_images'][member])
                                            
                                        #print('cluster', new_cluster)
                                        if agent.face_tree_cluster is not None:
                                            d = []
                                            new_cluster_ = []
                                            for emb_ in new_cluster:
                                                distance, index = agent.face_tree_cluster.query(emb_)
                                                if distance > 0.: # Sample with distance 0.0 is invalid
                                                    #print('name and distance', self.face_names[index], distance)
                                                    d.append(distance)
                                                    new_cluster_.append(emb_)
                                            #print('distance stats', len(d), d)
                                            if len(d) >= FACE_EMBEDDING_SAMPLE_SIZE:
                                                d_mean = np.mean(d)
                                                if d_mean < FACE_EMBEDDING_THRESHOLD_LOW:
                                                    # The face is already registered
                                                    info('Face already registered, d_mean: {}, d: {}'.format(d_mean, d))
                                                    pass
                                                else:
                                                    # A new face is found
                                                    agent.register_new_cluster(new_cluster_, face_images)
                                        else:
                                            # The tree is empty, register this cluster
                                            agent.register_new_cluster(new_cluster, face_images)
                                    #print('end of one cluster iteration', ci)
                                #print('exit cluster iterations')

                            agent.check_update_tree()
                            agent.t_update_face_tree = 0

                else:
                    warn('The task is invalid: {}'.format(task))

                self.in_queue.task_done()
            
            except:
                self._on_crash()

class FaceDetectionThread(ThreadBase):
    """ The singleton that manages all detection """

    def _init_derived(self):
        self.tree_queue = Queue()
        
    def run(self):
        while True:
            if self.is_crashed:
                # The thread is broken; do nothing
                time.sleep(1)
                continue

            if not self.is_standby:
                # Do initialization here...
                # Initialize required models and sessions
                info('THREAD INITIALIZATION: {}'.format(threading.current_thread()))
                self.is_standby = True
                continue

            try:
                # Get task from input queue
                t_now = time.time() * 1000

                task = self.in_queue.get() # queue::get() is blocking by default

                if time.time() > task.t_expiration:
                    # Threads are busy and fail to process the task in time
                    # Something must be put in output queue for calling thread is waiting with get()
                    task.params['output_holder'].put_nowait(None)
                else:
                    #print(threading.current_thread(), 'queue to exec', time.time()*1000 - task.t_queued)
                    #print(task.params)
                    agent_id = task.params['agent']['agentId']

                    # Lazy detection mechanism
                    agstate = None
                    use_last_predictions = False
                    if agent_id in FaceApplications().agent_state:
                        agstate = FaceApplications().agent_state[agent_id]
                        t_detection = agstate['t_detection']
                        if t_now-t_detection < LAZY_DETECTION:
                            # Last detection results are still fresh and can be responded directly
                            use_last_predictions = True
                            predictions = copy.deepcopy(agstate['predictions'])
                            if 'identities' in predictions: predictions.pop('identities', None)
                            debug('Use lazy detection, {}, {}'.format(t_now-t_detection, LAZY_DETECTION))
                            #debug('Check lazy detection, t_detection: {}, t_recognition: {}, interval_detection: {}, interval_recognition: {}'.format(t_now-t_detection, t_now-t_recognition, LAZY_DETECTION, LAZY_RECOGNITION))

                    if not use_last_predictions:
                        predictions = self.detect(task.img, task.params)

                    # Lazy recognition mechanism
                    use_last_recognition = False
                    if agstate is not None:
                        if 'identities' in agstate['predictions'] and len(predictions['rectangles']):
                            if 'service' in task.params and 'mode' in task.params['service'] and task.params['service']['mode']=='register':
                                # Do not use lazy recognition mechanism when registering a face
                                pass
                            else:
                                t_recognition = agstate['t_recognition']
                                #debug('Check lazy recognition, t_recognition: {}, interval_recognition: {}'.format(t_now-t_recognition, LAZY_RECOGNITION))
                                if t_now-t_recognition < LAZY_RECOGNITION: # Check if face recognition results are outdated
                                    confidence_decay = (t_now-t_recognition) / (LAZY_RECOGNITION)
                                    len_frame_now = len(predictions['rectangles'])
                                    len_frame_last = len(agstate['predictions']['rectangles'])
                                    frame_id = []
                                    frame_id = frame_id + [0]*len_frame_now
                                    frame_id = frame_id + [1]*len_frame_last
                                    rectangles = predictions['rectangles'] + agstate['predictions']['rectangles']

                                    # Calculate IOU of rectangles to determine if faces in 2 frames belong to a single identity
                                    clusters = imutil.group_rectangles_miniou(rectangles, threshold=0.5)
                                    #debug('Check IOU, clusters: {}, frame_id: {}'.format(clusters, frame_id))
                                    name = [''] * len_frame_now
                                    confidence = [0.] * len_frame_now
                                    #debug('Initialize name and confidence {} {} {}'.format(len_frame_now, name, confidence))
                                    for pair in clusters:
                                        if len(pair)==2: # Only one-one overlapping is considered same face
                                            index_now = pair[0]
                                            index_last = pair[1]
                                            if frame_id[index_now]!=frame_id[index_last]: # Overlapped rectangles not in the same frame
                                                # Rectangles IOU overlapped
                                                index_last = pair[1] - len_frame_now
                                                name[index_now] = agstate['predictions']['identities']['name'][index_last]
                                                confidence[index_now] = agstate['predictions']['identities']['confidence'][index_last] * confidence_decay
                                                if len(name[index_now]): use_last_recognition = True

                                    if use_last_recognition:
                                        predictions['identities'] = {
                                            'name': name,
                                            'confidence': confidence,
                                        }
                                        if 'recognition' in predictions:
                                            # Remove 'recognition' in predictions to skip face recognition computation
                                            predictions.pop('recognition', None)
                                        debug('Use lazy recognition, {}'.format(predictions['identities']))
                    
                    if 'recognition' in predictions and len(predictions['recognition']['rectangles']):
                        #print('embeddings', predictions['embeddings'])
                        names_, confidences_ = FaceApplications().query_embedding(task.params['agent']['agentId'], predictions['recognition']['embeddings'])

                        # Register faces only on request to prevent false positive new face registering
                        task_embeddings = {
                            'agent': task.params['agent'],
                            'names': names_,
                            'confidences': confidences_,
                            'recognition': predictions['recognition'],
                        }
                        FaceApplications().register_embedding(task_embeddings)

                        n_rect = len(predictions['rectangles'])
                        names = [''] * n_rect
                        confidences = [0.] * n_rect
                        for s_index, a_index in enumerate(predictions['recognition']['index']):
                            names[a_index] = names_[s_index]
                            confidences[a_index] = confidences_[s_index]
                        predictions['identities'] = {
                            'name': names,
                            'confidence': confidences,
                        }
                        #print(predictions)
                        predictions.pop('recognition', None) # embeddings are not supposed to be returned to client

                    if not use_last_predictions and not use_last_recognition:
                        FaceApplications().on_predictions(task.params['agent'], t_now, predictions)

                    debug('FaceDetectionThread task done, elapsed: {}, thread: {}'.format(time.time()*1000-task.t_queued, threading.current_thread()))
                    task.params['output_holder'].put_nowait({'predictions': predictions})

                self.in_queue.task_done()

            except:
                self._on_crash()

    def detect(self, img, params):
        """
        Input: pixel array, shape (h, w)
        Ouput: list of rectangles of objects, shape (count, y, x, h, w)
        """
        predictions = {
            'rectangles': [],
            'confidences': [],
            'sort_index': [],
            'landmarks': [],
            'emotions': [],
            'recognition': {
                'index': [],
                'rectangles': [],
                'embeddings': [],
                'images': [],
            },
            'timing': {},
        }
        if 'service' in params:
            #{'type': 'face', 'options': {'resultsLimit': 5}, 'model': '12-net'}
            service = params['service']
            model = service['model']

            mode = ''
            if 'mode' in service: mode = service['mode']

            # Limit image size for performance
            #print('service', service)
            t_ = time.time()

            # Prepare parameters
            res_cap = 448
            factor = 0.6
            interp = cv2.INTER_NEAREST
            if 'options' in service:
                options = service['options']
                if 'res_cap' in options:
                    res_cap = int(options['res_cap'])
                if 'factor' in options:
                    factor = float(options['factor'])
                if 'interp' in options:
                    if options['interp']=='LINEAR':
                        interp = cv2.INTER_LINEAR
                    elif options['interp']=='AREA':
                        interp = cv2.INTER_AREA
            if factor > 0.9: factor = 0.9
            elif factor < 0.45: factor = 0.45
            # For performance reason, resolution is hard-capped at 800, which is suitable for most applications
            if res_cap > 800: res_cap = 800
            elif res_cap <= 0: res_cap = 448 # In case client fail to initialize res_cap yet included options in request
            #print('options', factor, interp)
            
            # This is a safe guard to avoid very large image,
            # for best performance, client is responsible to scale the image before upload
            resized, scale_factor = imutil.fit_resize(img, maxsize=(res_cap, res_cap), interpolation=interp)
            scale_factor = 1. / scale_factor
            predictions['timing']['fit_resize'] = (time.time() - t_) * 1000

            #print('mean', np.mean(img), np.mean(resized))

            #time_start = time.time()
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # MTCNN face detection
            t_ = time.time()
            detector = FaceApplications().get_detector_create()
            pnet = detector['pnet']
            rnet = detector['rnet']
            onet = detector['onet']
            extents, landmarks = FaceDetector.detect_face(resized, 40, pnet, rnet, onet, threshold=[0.6, 0.7, 0.9], factor=factor, interpolation=interp)
            predictions['timing']['mtcnn'] = (time.time() - t_) * 1000

            # For testing, detect second time with ROIs (like in tracking mode)
            """
            # Pending; this is difficult to implement for multi-threading.
            if len(extents):
                t_ = time.time()
                height, width, *_ = resized.shape
                #print()
                #print('shape', resized.shape)
                #print('extents', extents)
                # Expand extents
                for i, e in enumerate(extents):
                    w = e[2] - e[0]
                    h = e[3] - e[1]
                    extents[i] += np.array((-w, -h, w, h, 0), dtype=np.float)
                    e = extents[i]
                    extents[i] = np.array((max(e[0], 0), max(e[1], 0), min(e[2], width-1), min(e[3], height-1), e[4]), dtype=np.float)
                #print('expand', extents)
                # Group overlapped extents
                for iteration in range(16):
                    no_overlap = True
                    for i1, e1 in enumerate(extents):
                        if e1[4]==0: continue
                        for i2, e2 in enumerate(extents):
                            if e2[4]==0: continue
                            if i1!=i2:
                                if e2[0]>e1[2] or e2[1]>e1[3] or e2[2]<e1[0] or e2[3]<e1[1]:
                                    pass
                                else:
                                    no_overlap = False
                                    extents[i1] = np.array((min((e1[0], e2[0])), min((e1[1], e2[1])), max((e1[2], e2[2])), max((e1[3], e2[3])), 1), dtype=np.float)
                                    extents[i2] = np.array((0, 0, 0, 0, 0), dtype=np.float)
                    if no_overlap: break
                #print(type(extents[0]))
                #print('group', extents)
                predictions['timing']['prepare_roi'] = (time.time() - t_) * 1000
                t_ = time.time()
                for i, e in enumerate(extents):
                    if e[4] > 0:
                        e = e.astype(dtype=np.int)
                        roi = resized[e[1]:e[3], e[0]:e[2], :]
                        _extents, _landmarks = FaceDetector.detect_face(roi, 40, pnet, rnet, onet, threshold=[0.6, 0.7, 0.9], factor=factor, interpolation=interp)
                        #print()
                        #print(i, roi.shape)
                        #print(_extents)
                predictions['timing']['mtcnn_roi'] = (time.time() - t_) * 1000
                predictions['timing']['mtcnn_roi_total'] = predictions['timing']['prepare_roi'] + predictions['timing']['mtcnn_roi']
            """

            if len(landmarks):
                landmarks = np.array(landmarks) * scale_factor
            """if len(extents):
                _ = np.array(landmarks) * scale_factor
                _ = _.reshape(2, -1)
                _ = np.transpose(_)
                _ = _.reshape((-1, len(extents), 2))
                landmarks = np.swapaxes(_, 0, 1)"""

            if model=='a-emoc':
                facelist = np.zeros((len(extents), 48, 48), dtype=np.float)
                predictions['timing']['emoc_prepare'] = 0
            elif model=='fnet':
                aligned_face_list = np.zeros((len(extents), 160, 160, 3), dtype=np.float)
                pass
            
            #print()
            #print(extents)
            sorting_index = {}
            for i, e in enumerate(extents):
                #e_ = (Rectangle(r_[0], r_[1], r_[2], r_[3]).to_extent().eval() * scale_factor).astype(dtype=np.int).tolist()
                r_ = (np.array([e[0], e[1], e[2]-e[0], e[3]-e[1]])*scale_factor).astype(dtype=np.int).tolist()
                #print('extents', i, e, r_)
                predictions['rectangles'].append(r_)
                predictions['confidences'].append(e[4])
                plist_ = landmarks[i].astype(dtype=np.int).tolist()
                predictions['landmarks'].append(plist_)
                if model=='a-emoc':
                    t_ = time.time()
                    #(x, y, w, h) = imutil.rect_fit_ar(r_, [0, 0, img.shape[1], img.shape[0]], 1., crop=False)
                    (x, y, w, h) = imutil.rect_fit_points(landmarks[i], )
                    r_ = np.array([x, y, w, h]).astype(dtype=np.int).tolist()
                    
                    # Return landmarks-corrected reactangles instead of MTCNN rectangle.
                    # The rectangles are used to subsample faces for emotion recognition.
                    predictions['rectangles'][-1] = r_

                    (x, y, w, h) = r_
                    face = np.zeros((h, w, 3), dtype=np.float)
                    y_ = 0
                    x_ = 0
                    if y + h > img.shape[0]:
                        h = img.shape[0] - y
                    elif y < 0:
                        y_ = -y
                        y = 0
                        h = h - y_
                    if x + w > img.shape[1]:
                        w = img.shape[1] - x
                    elif x < 0:
                        x_ = -x
                        x = 0
                        w = w - x_
                    face[y_:y_+h, x_:x_+w, :] = img[y:y+h, x:x+w, :]

                    face = cv2.resize(face, (48, 48), interpolation = interp)
                    face = (face[:, :, 0:1] * 0.2126 + face[:, :, 1:2] * 0.7152 + face[:, :, 2:3] * 0.0722).reshape((48, 48))
                    #face_write = (face * 255.).astype(np.uint8)
                    #cv2.imwrite('./face'+str(i).zfill(3)+'.jpg', face_write)
                    facelist[i:i+1, :, :] = face
                    predictions['timing']['emoc_prepare'] += (time.time() - t_) * 1000
                elif model=='fnet':
                    aligned = FaceApplications.align_face_fnet(img, landmarks[i])
                    aligned_face_list[i, :, :, :] = aligned
                    #print('aligned', aligned.shape)
                
                if e[0] <= 0 or e[1] <= 0 or e[2] >= img.shape[1]-1 or e[3] >= img.shape[0]-1:
                    sorting_index[i] = 0
                else:
                    sorting_index[i] = e[4] * r_[2] * r_[3]
            
            predictions['sort_index'] = sorting_index

            # Sort faces by sorting_index and select first N faces for computation intensive operations, e.g., facenet embedding extraction
            better_faces = sorted(sorting_index, key=sorting_index.get, reverse=True)
            better_faces = better_faces[0:FACE_RECOGNITION_CONCURRENT]
            #print('sort', sorting_index, better_faces)
            better_aligned_face_list = np.zeros((len(better_faces), 160, 160, 3), dtype=np.float)
            better_rectangles = []
            for better_face_index in range(len(better_faces)):
                better_aligned_face_list[better_face_index, :, :, :] = aligned_face_list[better_faces[better_face_index], :, :, :]
                better_rectangles.append(predictions['rectangles'][better_face_index])

            if model=='a-emoc':
                t_ = time.time()
                emoc_ = FaceApplications().get_emotion_classifier_create()
                emotions = emoc_.predict(facelist)
                predictions['emotions'] = emotions
                predictions['timing']['emoc'] = (time.time() - t_) * 1000
            elif model=='fnet':
                #print('aligned_face_list', aligned_face_list.shape)
                t_ = time.time()
                facenet_ = FaceApplications().get_facenet_create()
                #with facenet_.as_default():
                if True:
                    """images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                    feed_dict = { images_placeholder: better_aligned_face_list, phase_train_placeholder: False }
                    emb = facenet_.run(embeddings, feed_dict=feed_dict)"""
                    emb = facenet_(better_aligned_face_list)
                    #predictions['embeddings'] = np.concatenate((predictions['embeddings'], emb))
                    predictions['recognition']['index'] = better_faces
                    predictions['recognition']['rectangles'] = predictions['recognition']['rectangles'] + better_rectangles
                    predictions['recognition']['embeddings'] = predictions['recognition']['embeddings'] + emb.tolist()
                    predictions['recognition']['images'] = predictions['recognition']['images'] + better_aligned_face_list.tolist()
                predictions['recognition']['mode'] = mode
                t_ = time.time() - t_
                #print('facenet forward', emb.shape, t_*1000)
                #print()

        else:
            # Nothing is requested; useful for measuring overhead
            pass

        return predictions

    def align_dataset(self, directory=None):
        if directory is None:
            directory = '../data/face/fer2013_raw'

        stats = {
            'count': 0,
            'left': [0., 0.],
            'right': [0., 0.],
            'low': [0., 0.],
        }

        while True:
            f = dirwalker().get_a_file(directory=directory, filters=['.jpg'])
            if f is None: break

            img = cv2.imread(f.path, 1)
            if img is None:break
            img = (img.astype(dtype=np.float32))/255

            detector = FaceApplications().get_detector_create()
            pnet = detector['pnet']
            rnet = detector['rnet']
            onet = detector['onet']
            extents, landmarks = FaceDetector.detect_face(img, 48, pnet, rnet, onet, threshold=[0.6, 0.7, 0.9], factor=0.5)

            if len(landmarks) and len(landmarks[0]):
                warpped = self.align_face(img, landmarks[0])

                outpath = f.path.replace('fer2013_raw', 'fer2013_aligned')
                if outpath != f.path:
                    print(outpath)
                    outdir = os.path.split(outpath)[0]
                    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(outpath, warpped*255)

                """# Statitics of positions of eyes and nose
                stats['count'] += 1
                count = stats['count']

                stats['left'][0] = stats['left'][0]*(count-1)/count + left_eye[0]/count
                stats['left'][1] = stats['left'][1]*(count-1)/count + left_eye[1]/count
                stats['right'][0] = stats['right'][0]*(count-1)/count + right_eye[0]/count
                stats['right'][1] = stats['right'][1]*(count-1)/count + right_eye[1]/count
                stats['low'][0] = stats['low'][0]*(count-1)/count + low_center[0]/count
                stats['low'][1] = stats['low'][1]*(count-1)/count + low_center[1]/count

                print(stats)"""

            else:
                pass

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class VisionMainThread(metaclass=Singleton):
    """
    The singleton that manages all computer vision cores of a specific function block, e.g., face application, object detection, scene segmentation....
    Each function block (a set of related applications) maintains its own thread with status, input queue, output queue...
    Each function block (a set of related applications) spawns multiple threads as 'core' that do the actual detection.
    """
    def __init__(self):
        pass

class FaceApplications(VisionMainThread):
    """ This is for backward compatibility and the interface MUST NOT be changed. """
    def __init__(self):
        # Tensorflow sessions are thread-safe and can be called from multple threads simultaneously
        # Call tf.Graph::finalize() to lock graphs
        self.__mtcnn = None
        self.__emoc = None
        self.__facenet = None
        self.get_detector_create()
        self.get_emotion_classifier_create()
        self.get_facenet_create()

        self.agent_state = {} # Persistent states for agents

        self.core_threads = []
        self.in_queue = Queue(maxsize=8)
        self.emb_queue = Queue(maxsize=32)
        self.io_queue = Queue(maxsize=64)

        self.face_tree = {}
        self.face_names_flatten = {}

        for i in range(2): # Spawn 2 threads
            t = FaceDetectionThread(self.in_queue)
            t.setDaemon(True)
            self.core_threads.append(t)
            t.start()

            """# Signal the thread to initialize
            signal_init = ThreadTask(None, None)
            signal_init.t_expiration = -1
            t.in_queue.put(signal_init)"""

        t = FaceEmbeddingThread(self.emb_queue)
        t.setDaemon(True)
        t.start()

        t = FileWritingThread(self.io_queue)
        t.setDaemon(True)
        t.start()

        while True:
            count = len(self.core_threads)
            for t in self.core_threads:
                if t.is_standby:
                    count -= 1
            if count==0:
                break
            time.sleep(1)

        # Wait for core initialization
        #for t in self.core_threads:
        #    t.join()

        info('FaceApplications STANDBY: {}'.format(threading.current_thread()))

    def get_detector_create(self):
        if self.__mtcnn is None:
            info('Loading MTCNN, thread: {}'.format(threading.current_thread()))
            self.__mtcnn = {}
            self.__mtcnn['session'] = tf.Session()
            #self.__pnet, self.__rnet, self.__onet = FaceDetector.create_mtcnn(self.__mtcnn, None)
            self.__mtcnn['pnet'], self.__mtcnn['rnet'], self.__mtcnn['onet'] = FaceDetector.create_mtcnn(self.__mtcnn['session'], None)
            info('MTCNN loaded, thread: {}'.format(threading.current_thread()))
        return self.__mtcnn

    def get_emotion_classifier_create(self):
        if self.__emoc is None:
            info('Loading emotion classifier, thread: {}'.format(threading.current_thread()))
            self.__emoc = EmotionClassifier()
            self.__emoc.build_network(None)
            info('Emotion classifier loaded, thread: {}'.format(threading.current_thread()))
        return self.__emoc

    def get_facenet_create(self):
        if self.__facenet is None:
            info('Loading FaceNet, thread: {}'.format(threading.current_thread()))
            #self.__facenet = facenet.load_model('../models/facenet/20170512-110547.pb') # InceptionResnet V1
            self.__facenet = facenet.load_model('../models/facenet/20180204-160909') # squeezenet
            info('FaceNet loaded, thread: {}'.format(threading.current_thread()))
        return self.__facenet

    def file_write(self, filepath, content):
        if not self.io_queue.full():
            io_task = ThreadTask()
            io_task.write = {
                'filepath': filepath,
                'content': content,
            }
            self.io_queue.put_nowait(io_task)
            return True
        else:
            warning('FileWritingThread is busy and discarding data... {} bytes'.format(len(content)))
            return False

    def register_embedding(self, task_embeddings):
        # Queue face embedding to be processed by FaceEmbeddingThread to find candidates to be registered to database of known faces
        if not self.emb_queue.full():
            emb_task = ThreadTask()
            emb_task.embeddings = task_embeddings
            self.emb_queue.put_nowait(emb_task)
        #print('register_embedding', self.emb_queue.qsize(), task_embeddings['agent'])

    def tree_updated(self, agent_id, tree, names_cluster):
        # Notified by FaceEmbeddingThread that search tree of known faces is updated
        # This is called from multiple threads. Fortunately, Python dict is thread-safe
        self.face_tree[agent_id] = tree
        self.face_names_flatten[agent_id] = []
        info('Tree updated, agent: {}, faces: {}'.format(agent_id, len(names_cluster)))
        for name, count in names_cluster:
            for _ in range(count):
                self.face_names_flatten[agent_id].append(name)

    def query_embedding(self, agent_id, embeddings):
        names = [''] * len(embeddings)
        confidences = np.zeros((len(embeddings),), dtype=np.float)

        if agent_id in self.face_tree:
            embeddings = sklearn.preprocessing.normalize(embeddings)
            for i, emb in enumerate(embeddings):
                distance, index = self.face_tree[agent_id].query(emb, FACE_EMBEDDING_SAMPLE_SIZE)

                # Find most frequent matched name
                name_freq = {}
                name_freq_max = 0
                #print('closet neighbors', index)
                for j, face_index in enumerate(index):
                    name = self.face_names_flatten[agent_id][face_index]
                    if name in name_freq:
                        name_freq[name] = name_freq[name] + 1
                    else:
                        name_freq[name] = 0
                    if name_freq[name] > name_freq_max:
                        name_freq_max = name_freq[name]
                        name_freq_name = name
                    #print('query_embedding, neighbor', name, distance[j])

                # At least FACE_EMBEDDING_SAMPLE_SIZE/2 samples are required to consider a positive match
                if name_freq_max*2 >= FACE_EMBEDDING_SAMPLE_SIZE:
                    d = []
                    for j, face_index in enumerate(index):
                        if self.face_names_flatten[agent_id][face_index]==name_freq_name:
                            d.append(distance[j])
                    d_mean = np.mean(d)
                    if d_mean < FACE_EMBEDDING_THRESHOLD_RECALL:
                        #print('similar face', distance, index, self.face_names_flatten[agent_id][index[0]])
                        confidences[i] = 1. - d_mean
                        names[i] = name_freq_name
        
        return (names, confidences)

    def detect(self, img, params):
        """ Queue the task """
        #print('detect', self.in_queue.qsize())
        if self.in_queue.full():
            return False
        else:
            t = ThreadTask(img=img, params=params)
            self.in_queue.put_nowait(t)
            return True

    def on_predictions(self, agent, t_detection, predictions):
        """ Save predictions in memory for lazy detection """
        agent_id = agent['agentId']
        if agent_id not in self.agent_state:
            self.agent_state[agent_id] = {
                't_detection': 0,
                't_recognition': 0,
                'predictions': None,
            }
        agstate = self.agent_state[agent_id]

        agstate['t_detection'] = t_detection
        agstate['predictions'] = predictions
        if 'identities' in predictions:
            agstate['t_recognition'] = t_detection
            agstate['identities'] = predictions['identities']
        else:
            agstate['t_recognition'] = 0
            agstate['predictions'].pop('identities', None)

    @staticmethod
    def align_face_fnet(img, landmarks):
        """ Shorthand for aling_face() for FaceNet 160x160x3 """
        return FaceApplications.align_face(img, landmarks, intensity=1., sz=160, ortho=True, expand=1.5, scale_limit=2.)

    @staticmethod
    def align_face(img, landmarks, intensity=0.5, sz=48, ortho=False, expand=1.0, scale_limit=0.):
        # Normalize face pose and position with landmarks of 5 points: eyes, nose, mouth corners

        # Convert landmarks to numpy array
        left_eye = np.array(landmarks[0], dtype=np.float)
        right_eye = np.array(landmarks[1], dtype=np.float)
        nose = np.array(landmarks[2], dtype=np.float)
        mouth = (np.array(landmarks[3], dtype=np.float) + np.array(landmarks[4], dtype=np.float))/2.
        low_center = (np.array(landmarks[2], dtype=np.float) + np.array(landmarks[3], dtype=np.float) + np.array(landmarks[4], dtype=np.float)) / 3.

        # Calculate correction vectors for vertical and horizontal axes
        eye_center = (right_eye+left_eye) / 2.
        vv = nose - eye_center
        vh = (right_eye - left_eye)

        if ortho:
            vv = np.roll(vh, 1)
            vv *= 17. / 24. * expand
            vv[0] *= -1
            vh *= expand

        if scale_limit > 0.:
            distance = (vh[0]*vh[0] + vh[1]*vh[1])/sz

        # Change length of correction vectors to be proportionate to face dimension
        vv_high = vv * 35. / 17.
        vv_low = vv * 13. / 17.

        # Calculate 4 corners for perspective transformation
        corners = np.array(
            [low_center - vv_high - vh,
            low_center - vv_high + vh,
            low_center + vv_low + vh,
            low_center + vv_low - vh,],
            dtype= np.float)

        """debug = np.array(
            [low_center - vv_high - vh,
            low_center - vv_high + vh,
            low_center + vv_low + vh,
            low_center + vv_low - vh,
            low_center,
            low_center + vh,
            low_center + vv,
            ],
            dtype= np.float)
        return debug"""

        # Perspective transformation
        rect = np.array(corners, dtype = "float32")
        dst = np.array([[0, 0], [sz-1, 0], [sz-1, sz-1], [0, sz-1]], dtype = "float32")
        rect = rect*intensity + dst*(1.-intensity)
        M = cv2.getPerspectiveTransform(rect, dst)

        return cv2.warpPerspective(img, M, (sz, sz), borderMode=cv2.BORDER_CONSTANT)

    def align_dataset(self, directory=None):
        pass

if __name__== "__main__":
    face_app = FaceApplications()
    face_app.detect()

    r_ = Rectangle(12, 24, 48, 96)
    print(r_.eval())
    e_ = r_.to_extent()
    print(e_.eval())
    r_ = e_.to_rectangle()
    print(r_.eval())

