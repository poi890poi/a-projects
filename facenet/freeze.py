"""Functions for building the face recognition network.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import Popen, PIPE
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
from tensorflow.python.training import training
import random
import re
from tensorflow.python.platform import gfile
from six import iteritems

from tensorflow.python.tools import inspect_checkpoint
from tensorflow.python.framework.graph_util import convert_variables_to_constants

def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    tf.reset_default_graph() # This line is required when multiple models are used,
                             # otherwise error 'NotFoundError: Key is_training not found in checkpoint'
                             # will be encountered when restoring checkpoints

    if os.path.isfile(model): # Load from .pb file
        g_ = tf.Graph()
        with g_.as_default():
            with gfile.FastGFile(model, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

            sess = tf.Session(graph=g_)
            for op in g_.get_operations():
                op_name = str(op.name)
                if not op_name.startswith('InceptionResnet'):
                    print(op_name)

            init = tf.global_variables_initializer()
            sess.run(init)

    else: # Load from checkpoint and meta files
        g_ = tf.Graph()
        with g_.as_default():
            sess = tf.Session(graph=g_)

            meta_file, ckpt_file = get_model_filenames(model)
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            saver = tf.train.import_meta_graph(os.path.join(model, meta_file))
            saver.restore(sess, os.path.join(model, ckpt_file))

            save_path = saver.save(sess, "../../models/facenet/squeeze/squeeze.ckpt")

            variable_names_blacklist = [
                    'squeezenet/conv1/BatchNorm/moving_mean',
                    'squeezenet/conv1/BatchNorm/moving_variance',
                    'squeezenet/Bottleneck/BatchNorm/moving_mean',
                    'squeezenet/Bottleneck/BatchNorm/moving_variance',
                ]
            for i in range(2, 10):
                i_str = str(i)
                variable_names_blacklist = variable_names_blacklist + [
                    'squeezenet/fire{}/squeeze/BatchNorm/moving_mean'.format(i_str),
                    'squeezenet/fire{}/squeeze/BatchNorm/moving_variance'.format(i_str),
                    'squeezenet/fire{}/expand/1x1/BatchNorm/moving_mean'.format(i_str),
                    'squeezenet/fire{}/expand/1x1/BatchNorm/moving_variance'.format(i_str),
                    'squeezenet/fire{}/expand/3x3/BatchNorm/moving_mean'.format(i_str),
                    'squeezenet/fire{}/expand/3x3/BatchNorm/moving_variance'.format(i_str),
                ]

            """for op in g_.get_operations():
                op_name = str(op.name)
                if 'moving' in op_name:
                    print(op_name)"""

            """for v_ in tf.global_variables():
                v_name = str(v_.name)
                print(v_name, (v_.initializer), )"""
                
            output_graph_def = convert_variables_to_constants(
                sess,
                sess.graph_def,
                ['embeddings'],
                variable_names_blacklist=variable_names_blacklist)

            tf.train.write_graph(output_graph_def, '.', '../../models/facenet/facenet-squeeze.pb', as_text=False)
            tf.train.write_graph(output_graph_def, '.', '../../models/facenet/facenet-squeeze.txt', as_text=True)

    #fnet = lambda faces : sess.run(('embeddings:0'), feed_dict={'input:0': faces})
    #fnet = lambda faces : sess.run(g_.get_tensor_by_name("embeddings:0"), feed_dict={'input:0': faces, 'phase_train:0': False})
    fnet = lambda faces : sess.run(('embeddings:0'), feed_dict={'input:0': faces, 'phase_train:0': False})
    g_.finalize()

    return fnet

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

"""inspect_checkpoint.print_tensors_in_checkpoint_file(
    file_name='../../models/facenet/20180204-160909/model-20180204-160909.ckpt-266000',
    tensor_name='',
    all_tensors=False,
    all_tensor_names=True)"""

load_model('../../models/facenet/20180204-160909')