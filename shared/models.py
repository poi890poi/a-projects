import tensorflow as tf
import numpy as np

class FaceCascade():
    def __init__(self, params=None):
        self.params = params
        self.model_dir = params['model_dir']
        self.log_dir = self.model_dir + '/log'
        self.ckpt_prefix = params['ckpt_prefix']
        self.model()

    def model(self):
        mode = self.params['mode']

        shape_raw = (48, 48, 3)
        shape_flat = (np.prod(shape_raw),)
        shape_image = (12, 12, 3)
        n_class = 2

        summarize = False
        is_train = False
        is_inference = False
        if mode=='INFERENCE':
            is_inference = True
        elif mode in ['TRAIN', 'TEST']:
            if mode=='TRAIN': is_train = True
            summarize = True
            batch_size = self.params['batch_size']
            epochs = self.params['epochs']
            steps = self.params['steps']
            learn_rate = self.params['learn_rate']

            """if tf.gfile.Exists(self.log_dir):
                tf.gfile.DeleteRecursively(self.log_dir)
            tf.gfile.MakeDirs(self.log_dir)"""
        else:
            raise(ValueError, 'mode must be either ["INFERENCE", "TRAIN", "TEST"]')

        self.sess = tf.InteractiveSession()

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, np.prod(shape_raw)], name='train_data')
            self.y_ = tf.placeholder(tf.float32, [None, n_class], name='labels')
        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(self.x, (-1,)+shape_raw)
            if summarize: tf.summary.image('input', image_shaped_input, batch_size)

        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

        def fully_connected(input, size):
            weights = tf.get_variable( 'weights', 
                shape = [input.get_shape()[1], size],
                initializer = tf.contrib.layers.xavier_initializer()
            )
            biases = tf.get_variable( 'biases',
                shape = [size],
                initializer = tf.constant_initializer(0.0)
            )
            if summarize:
                variable_summaries(weights)
                variable_summaries(biases)
            return tf.matmul(input, weights) + biases

        def fully_connected_relu(input, size):
            return tf.nn.relu(fully_connected(input, size))

        def conv_relu(input, kernel_size, depth):
            weights = tf.get_variable( 'weights', 
                shape = [kernel_size, kernel_size, input.get_shape()[3], depth],
                initializer = tf.contrib.layers.xavier_initializer()
            )
            biases = tf.get_variable( 'biases',
                shape = [depth],
                initializer = tf.constant_initializer(0.0)
            )
            if summarize:
                variable_summaries(weights)
                variable_summaries(biases)
            conv = tf.nn.conv2d(input, weights,
                strides = [1, 1, 1, 1], padding = 'SAME')

            """if is_train:
                # Batch  normalization
                mean, var = tf.nn.moments(conv, [0])
                conv = tf.nn.batch_normalization(conv, mean, var, offset=None, scale=None, variance_epsilon=1e-3)"""

            return tf.nn.relu(conv + biases)

        def pool(input, size, stride):
            return tf.nn.max_pool(
                input, 
                ksize = [1, size, size, 1], 
                strides = [1, stride, stride, 1], 
                padding = 'SAME'
            )

        with tf.variable_scope('12-net'):
            input_12 = tf.image.resize_bilinear(image_shaped_input, (12, 12))
            # Convolutions
            with tf.variable_scope('conv1'):
                conv12_1 = conv_relu(input_12, kernel_size=3, depth=16)
                pool12_1 = pool(conv12_1, size=3, stride=2)
            shape = pool12_1.get_shape().as_list()
            flatten12 = tf.reshape(pool12_1, [-1, shape[1] * shape[2] * shape[3]])
            with tf.variable_scope('fc1'):
                fc12_1 = fully_connected_relu(flatten12, size=16)
                fc_final = fc12_1

        """with tf.variable_scope('24-net'):
            input_24 = tf.image.resize_bilinear(image_shaped_input, (24, 24))
            # Convolutions
            with tf.variable_scope('conv1'):
                conv24_1 = conv_relu(input_24, kernel_size=5, depth=64)
                pool24_1 = pool(conv24_1, size=3, stride=2)
            shape = pool24_1.get_shape().as_list()
            flatten24 = tf.reshape(pool24_1, [-1, shape[1] * shape[2] * shape[3]])
            print('net-24 flatten', flatten24)
            with tf.variable_scope('fc1'):
                fc24_1 = fully_connected_relu(flatten24, size=128)
            
            fc24_concat = tf.concat([fc24_1, fc12_1], 1)
            print('net-24 fc', fc24_1)
            print('net-24 concat', fc24_concat)

        with tf.variable_scope('48-net'):
            # Convolutions
            with tf.variable_scope('conv1'):
                conv48_1 = conv_relu(image_shaped_input, kernel_size=5, depth=64)
                pool48_1 = pool(conv48_1, size=3, stride=2)
                # Normalize, region=9
            with tf.variable_scope('conv2'):
                conv48_2 = conv_relu(pool48_1, kernel_size=5, depth=64)
                # Normalize, region=9
                pool48_2 = pool(conv48_2, size=3, stride=2)
            shape = pool48_2.get_shape().as_list()
            flatten48 = tf.reshape(pool48_2, [-1, shape[1] * shape[2] * shape[3]])
            with tf.variable_scope('fc1'):
                fc48_1 = fully_connected_relu(flatten48, size=256)
            
            fc48_concat = tf.concat([fc48_1, fc24_concat], 1)
            print('net-48 concat', fc48_concat)"""

        if is_train:
            with tf.variable_scope('dropout'):
                self.keep_prob = tf.placeholder(tf.float32)
                tf.summary.scalar('dropout_keep_probability', self.keep_prob)
                fc_final = tf.nn.dropout(fc_final, self.keep_prob)
        with tf.variable_scope('out'):
            self.y = fully_connected(fc_final, size=n_class)

        if is_inference:
            self.y = tf.nn.softmax(self.y)

        if is_train:
            with tf.name_scope('cross_entropy'):
                # The raw formulation of cross-entropy,
                #
                # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
                #                               reduction_indices=[1]))
                #
                # can be numerically unstable.
                #
                # So here we use tf.nn.softmax_cross_entropy_with_logits on the
                # raw outputs of the nn_layer above, and then average across
                # the batch.
                diff = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y)
                with tf.name_scope('total'):
                    self.cross_entropy = tf.reduce_mean(diff)
            tf.summary.scalar('cross_entropy', self.cross_entropy)

            with tf.name_scope('train'):
                self.learning_rate = tf.placeholder(tf.float32, shape=[])
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(
                    self.cross_entropy)

        if summarize:
            print('TRAIN')
            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
                with tf.name_scope('accuracy'):
                    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

            # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.log_dir+'/train', self.sess.graph)
            self.test_writer = tf.summary.FileWriter(self.log_dir+'/test')
            tf.global_variables_initializer().run()

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver(filename=self.ckpt_prefix)

        if is_inference:
            # Restore variables from disk.
            self.saver.restore(self.sess, self.ckpt_prefix)
            print('Model restored.', self.ckpt_prefix)

class SimpleClassifier():
    def __init__(self, model, train_id, params=None):
        self.model = model
        self.train_id = train_id

        self.learn_rate = 0.001
        if params and 'learn_rate' in params:
            self.learn_rate = params['learn_rate']

        self.layers = list()
        self.estimator_train = None
        self.estimator_eval = None
        self.estimator_predict = None

    def model_fn(self, features, labels, mode, params=None):
        #input_layer = tf.reshape(features['x'], [-1, 64, 96, 3])
        input_layer = features['x']

        self.layers = list()
        if self.model=='darknet19':
            self.darknet19(input_layer, labels, mode)
        elif self.model=='darknet':
            self.darknet(input_layer, labels, mode)
        elif self.model=='afanet':
            self.afanet(input_layer, labels, mode)
        elif self.model=='afanet11':
            self.afanet11(input_layer, labels, mode)
        elif self.model=='afayolo':
            self.afayolo(input_layer, labels, mode)
        else:
            raise ValueError('Unrecognized model name')

        logits = self.layers[-1]

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            'classes': tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            'probabilities': tf.nn.softmax(logits, name='softmax')
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        """
        #loss = tf.reduce_mean(tf.square(expected - net))
        print('out_weights', params['out_weights'])
        print('labels', labels)
        print('logits', logits)
        loss = tf.abs(labels - tf.cast(logits, tf.float64))*params['out_weights']
        print('loss', loss)
        loss = tf.reduce_mean(tf.abs(labels - tf.cast(logits, tf.float64))*params['out_weights'])
        #loss = tf.reduce_mean(tf.abs(labels - tf.cast(logits, tf.float64)))
        #loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)"""

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                labels=labels, predictions=predictions['classes'])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def get_estimator(self, out_weights=None, model_dir=''):
        # Create the Estimator
        if model_dir=='':
            if self.train_id:
                model_dir = '../models/' + self.model + '_' + self.train_id
            else:
                model_dir = '../models/' + self.model
        return tf.estimator.Estimator(
            model_fn=self.model_fn, params={'out_weights': out_weights}, model_dir=model_dir)

    def afayolo(self, input_layer, labels, mode):
        """
         0 conv     16  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  16
         1 max          2 x 2 / 2   416 x 416 x  16   ->   208 x 208 x  16
         2 conv     32  3 x 3 / 1   208 x 208 x  16   ->   208 x 208 x  32
         3 max          2 x 2 / 2   208 x 208 x  32   ->   104 x 104 x  32
         4 conv     64  3 x 3 / 1   104 x 104 x  32   ->   104 x 104 x  64
         5 max          2 x 2 / 2   104 x 104 x  64   ->    52 x  52 x  64
         6 conv    128  3 x 3 / 1    52 x  52 x  64   ->    52 x  52 x 128
         7 max          2 x 2 / 2    52 x  52 x 128   ->    26 x  26 x 128
         8 conv    256  3 x 3 / 1    26 x  26 x 128   ->    26 x  26 x 256
         9 max          2 x 2 / 2    26 x  26 x 256   ->    13 x  13 x 256
        10 conv    512  3 x 3 / 1    13 x  13 x 256   ->    13 x  13 x 512
        11 max          2 x 2 / 1    13 x  13 x 512   ->    13 x  13 x 512
        12 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
        13 conv   1024  3 x 3 / 1    13 x  13 x1024   ->    13 x  13 x1024
        14 conv    125  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 125
        """
        initial_filters = 16
        for i in range(5):
            self.layers.append(self.conv2d(input_layer, initial_filters, kernel_size=[3, 3], strides=[1, 1], activation='leaky_relu'))
            self.layers.append(self.pool2d(None))
            initial_filters *= 2

        self.layers.append(self.conv2d(input_layer, initial_filters, kernel_size=[3, 3], strides=[1, 1], activation='leaky_relu'))
        self.layers.append(self.conv2d(input_layer, initial_filters, kernel_size=[3, 3], strides=[1, 1], activation='leaky_relu'))
        self.layers.append(self.conv2d(None, 50, kernel_size=[1, 1], activation='leaky_relu'))

        for i, layer in enumerate(self.layers):
            print(i, layer)

    def afanet(self, input_layer, labels, mode):
        initial_filters = 16
        for i in range(5):
            self.layers.append(self.conv2d(input_layer, initial_filters, kernel_size=[3, 3], strides=[1, 1], activation='leaky_relu'))
            self.layers.append(self.pool2d(None))
            initial_filters *= 2

        self.layers.append(self.conv2d(input_layer, initial_filters, kernel_size=[3, 3], strides=[1, 1], activation='leaky_relu'))
        self.layers.append(self.conv2d(input_layer, initial_filters, kernel_size=[3, 3], strides=[1, 1], activation='leaky_relu'))

        self.layers.append(self.conv2d(None, 5, kernel_size=[1, 1], activation='leaky_relu'))
        self.layers.append(tf.reduce_mean(self.layers[-1], [1, 2], name='avg_pool'))

        for i, layer in enumerate(self.layers):
            print(i, layer)

    def afanet11(self, input_layer, labels, mode):
        self.layers.append(self.conv2d(input_layer, 32, kernel_size=[5, 5], strides=[2, 2], activation='leaky_relu'))
        self.layers.append(self.pool2d(None))
        self.layers.append(self.conv2d(None, 64, activation='leaky_relu'))
        self.layers.append(self.pool2d(None))

        self.layers.append(self.conv2d(None, 32, kernel_size=[1, 1], activation='leaky_relu'))
        self.layers.append(self.conv2d(None, 64, activation='leaky_relu'))
        self.layers.append(self.conv2d(None, 128, activation='leaky_relu'))
        self.layers.append(self.pool2d(None))

        self.layers.append(self.conv2d(None, 64, kernel_size=[1, 1], activation='leaky_relu'))
        self.layers.append(self.conv2d(None, 128, activation='leaky_relu'))

        self.layers.append(self.conv2d(None, 100, kernel_size=[1, 1], activation='leaky_relu'))
        self.layers.append(tf.reduce_mean(self.layers[-1], [1, 2], name='avg_pool'))
        
        for i, layer in enumerate(self.layers):
            print(i, layer)

    def darknet(self, input_layer, labels, mode):
        initial_filters = 16
        self.layers.append(self.conv2d(input_layer, initial_filters, kernel_size=[3, 3], strides=[1, 1], activation='leaky_relu'))
        self.layers.append(self.pool2d(None))
        initial_filters *= 2
        for i in range(3):
            self.layers.append(self.conv2d(None, initial_filters, kernel_size=[3, 3], strides=[1, 1], activation='leaky_relu'))
            self.layers.append(self.pool2d(None))
            initial_filters *= 2
        self.layers.append(self.conv2d(None, initial_filters, kernel_size=[3, 3], strides=[1, 1], activation='leaky_relu'))
        self.layers.append(self.dropout(None, rate=0.5))
        self.layers.append(self.conv2d(None, 200, kernel_size=[1, 1], strides=[1, 1], activation='leaky_relu'))
        self.layers.append(tf.reduce_mean(self.layers[-1], [1, 2], name='avg_pool'))
        
        for i, layer in enumerate(self.layers):
            print(i, layer)

    def darknet19(self, input_layer, labels, mode):
        '''
         0 conv     64  7 x 7 / 2   224 x 224 x   3   ->   112 x 112 x  64
         1 max          2 x 2 / 2   112 x 112 x  64   ->    56 x  56 x  64
         2 conv    192  3 x 3 / 1    56 x  56 x  64   ->    56 x  56 x 192
         3 max          2 x 2 / 2    56 x  56 x 192   ->    28 x  28 x 192
         4 conv    128  1 x 1 / 1    28 x  28 x 192   ->    28 x  28 x 128
         5 conv    256  3 x 3 / 1    28 x  28 x 128   ->    28 x  28 x 256
         6 conv    256  1 x 1 / 1    28 x  28 x 256   ->    28 x  28 x 256
         7 conv    512  3 x 3 / 1    28 x  28 x 256   ->    28 x  28 x 512
         8 max          2 x 2 / 2    28 x  28 x 512   ->    14 x  14 x 512
         9 conv    256  1 x 1 / 1    14 x  14 x 512   ->    14 x  14 x 256
        10 conv    512  3 x 3 / 1    14 x  14 x 256   ->    14 x  14 x 512
        11 conv    256  1 x 1 / 1    14 x  14 x 512   ->    14 x  14 x 256
        12 conv    512  3 x 3 / 1    14 x  14 x 256   ->    14 x  14 x 512
        13 conv    256  1 x 1 / 1    14 x  14 x 512   ->    14 x  14 x 256
        14 conv    512  3 x 3 / 1    14 x  14 x 256   ->    14 x  14 x 512
        15 conv    256  1 x 1 / 1    14 x  14 x 512   ->    14 x  14 x 256
        16 conv    512  3 x 3 / 1    14 x  14 x 256   ->    14 x  14 x 512
        17 conv    512  1 x 1 / 1    14 x  14 x 512   ->    14 x  14 x 512
        18 conv   1024  3 x 3 / 1    14 x  14 x 512   ->    14 x  14 x1024
        19 max          2 x 2 / 2    14 x  14 x1024   ->     7 x   7 x1024
        20 conv    512  1 x 1 / 1     7 x   7 x1024   ->     7 x   7 x 512
        21 conv   1024  3 x 3 / 1     7 x   7 x 512   ->     7 x   7 x1024
        22 conv    512  1 x 1 / 1     7 x   7 x1024   ->     7 x   7 x 512
        23 conv   1024  3 x 3 / 1     7 x   7 x 512   ->     7 x   7 x1024
        24 conv   1000  1 x 1 / 1     7 x   7 x1024   ->     7 x   7 x1000
        25 avg                        7 x   7 x1000   ->  1000
        26 softmax                                        1000
        27 cost                                           1000
        '''

        self.layers.append(self.conv2d(input_layer, 64, kernel_size=[7, 7], strides=[2, 2], activation='leaky_relu'))
        self.layers.append(self.pool2d(None))
        self.layers.append(self.conv2d(None, 192, activation='leaky_relu'))
        self.layers.append(self.pool2d(None))

        self.layers.append(self.conv2d(None, 128, kernel_size=[1, 1], activation='leaky_relu'))
        self.layers.append(self.conv2d(None, 256, activation='leaky_relu'))
        self.layers.append(self.conv2d(None, 256, kernel_size=[1, 1], activation='leaky_relu'))
        self.layers.append(self.conv2d(None, 512, activation='leaky_relu'))
        self.layers.append(self.pool2d(None))

        for i in range(4):
            self.layers.append(self.conv2d(None, 256, kernel_size=[1, 1], activation='leaky_relu'))
            self.layers.append(self.conv2d(None, 512, activation='leaky_relu'))
        self.layers.append(self.conv2d(None, 512, kernel_size=[1, 1], activation='leaky_relu'))
        self.layers.append(self.conv2d(None, 1024, activation='leaky_relu'))
        self.layers.append(self.pool2d(None))

        for i in range(2):
            self.layers.append(self.conv2d(None, 512, kernel_size=[1, 1], activation='leaky_relu'))
            self.layers.append(self.conv2d(None, 1024, activation='leaky_relu'))

        self.layers.append(self.conv2d(None, 1000, kernel_size=[1, 1], activation='leaky_relu'))
        print(self.layers[-1].shape)
        #self.layers.append(tf.nn.avg_pool(self.layers[-1], [1, 1, 7, 7], [1, 1, 1, 1], padding='SAME', name='avg_pool'))
        self.layers.append(tf.reduce_mean(self.layers[-1], [1, 2], name='avg_pool'))
        
        for i, layer in enumerate(self.layers):
            print(i, layer)

    def get_scope_name(self, index, suffix):
        return str(index).zfill(2)+'.'+suffix
    
    def conv2d(self, input_layer, num_filters, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu', normalize=True, name=''):
        if input_layer==None:
            input_layer = self.layers[-1]
        if name=='':
            name = self.get_scope_name(len(self.layers), 'conv')
        act_func = tf.nn.relu
        if activation=='leaky_relu':
            act_func = tf.nn.leaky_relu
        conv = tf.layers.conv2d(
            inputs=input_layer,
            filters=num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=act_func,
            name=name)
        if normalize:
            conv = tf.layers.batch_normalization(conv, name=name)
        return conv

    def pool2d(self, inputs, size=[2,2], strides=2, name=''):
        if inputs==None:
            inputs = self.layers[-1]
        if name=='':
            name = self.get_scope_name(len(self.layers), 'max')
        return tf.layers.max_pooling2d(inputs=inputs, pool_size=size, strides=strides, name=name)

    def dropout(self, inputs, rate=0.5, name=''):
        if inputs==None:
            inputs = self.layers[-1]
        dropout = tf.layers.dropout(
            inputs=inputs, rate=rate, name=name)
        return dropout

    def dense(self, inputs, units, activation='linear', dropout=-1, name=''):
        if inputs==None:
            inputs = self.layers[-1]
        if name=='':
            name = self.get_scope_name(len(self.layers), 'dense')
        act_func = None
        print(inputs.get_shape()[-3:], np.prod(inputs.get_shape()[-3:]))
        flatten = tf.reshape(inputs, [-1, np.prod(inputs.get_shape()[-3:])])
        dense = tf.layers.dense(inputs=flatten, units=units, activation=act_func, name=name)
        if dropout>0:
            dense = tf.layers.dropout(
                inputs=dense, rate=dropout, training=mode == tf.estimator.ModeKeys.TRAIN, name=name)
        return dense

