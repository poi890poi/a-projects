from shared.utilities import *
from shared.models import *
from shared.dataset import *
from face.dlibdetect import FaceDetector

import os.path
import pprint
import time

import cv2
import numpy as np
import tensorflow as tf

shape_image = (12, 12, 3)
shape_flat = (np.prod(shape_image),)
n_class = 2

def get_data():
    model_dir = '../models/cascade/preview'
    val_dir = os.path.join(model_dir, 'val')
    DataUtilities.prepare_dir(model_dir, 'val', empty=True)
    train_dir = os.path.join(model_dir, 'train')
    DataUtilities.prepare_dir(model_dir, 'train', empty=True)

    # Hyperparameters
    prep_denoise = False
    prep_equalize = False

    train_count = 200
    val_count = 200
    total_count = train_count + val_count

    train_positive_count = train_count//2
    train_negative_count = train_count - train_positive_count
    val_positive_count = val_count//2
    val_negative_count = val_count - val_positive_count
    positive_count = train_positive_count + val_positive_count
    negative_count = train_negative_count + val_negative_count

    a = np.zeros(shape=(total_count,)+shape_image, dtype=np.float32) # Data
    b = np.full(shape=[total_count, n_class], fill_value=-1, dtype=np.int) # Labels
    c = np.zeros(shape=[total_count, n_class], dtype=np.int) # Label assignment
    c[:] = [1, 0]
    c[0:train_positive_count] = [0, 1]
    c[train_count:train_count+val_positive_count] = [0, 1]

    train_data = a[0:train_count, :, :, :]
    train_labels = b[0:train_count, :]
    val_data = a[train_count:total_count, :, :, :]
    val_labels = b[train_count:total_count, :]

    # Load positive data from dataset
    data_dir = '../data/face/sof/images'
    dset = SoF()
    dset.load_annotations('../data/face/sof/images/metadata.mat', 'sof')
    count = positive_count
    face_index = 0
    while count:
        anno = dset.get_face(face_index)
        face_index += 1
        if anno is None:
            face_index = 0
            continue
        #print(anno)
        filename = '_'.join((anno['file_prefix'], 'e0_nl_o'))
        filename = '.'.join((filename, 'jpg'))
        imgpath = os.path.normpath(os.path.join(data_dir, filename))

        img = cv2.imread(imgpath, 1)
        if img is None:
            continue
        src_shape = img.shape
        height, width, *rest = img.shape
        (x, y, w, h) = ImageUtilities.rect_fit_ar(anno['rect'].astype(dtype=np.int), [0, 0, width, height], 1, mrate=1.)
        if w>0 and h>0:
            pass
        else:
            continue

        #face = img[y:y+h, x:x+w, :]
        face = ImageUtilities.transform_crop((x, y, w, h), img, r_intensity=1.0, p_intensity=0.)
        face = imresize(face, [12, 12])
        face = ImageUtilities.preprocess(face, convert_gray=None, equalize=prep_equalize, denoise=prep_denoise)
        face = np.array(face, dtype=np.float32)/255

        for i in range(len(b)):
            if np.array_equal(c[i], [0, 1]) and not np.array_equal(b[i], c[i]):
                break
        #print(i, 'positive', imgpath)
        a[i, :, :, :] = face
        b[i] = c[i]

        count -= 1

    # Load negative data
    count = negative_count
    while count:
        f = DirectoryWalker().get_a_file(directory='../data/face/processed/negative', filters=['.jpg'])
        img = cv2.imread(f.path, 1)
        height, width, *rest = img.shape
        crop = (height - width)//2
        img = img[crop:crop+width, :, :]
        img = imresize(img, [12, 12])
        img = ImageUtilities.preprocess(img, convert_gray=None, equalize=prep_equalize, denoise=prep_denoise)
        img = np.array(img, dtype=np.float32)/255

        for i in range(len(b)):
            if np.array_equal(c[i], [1, 0]) and not np.array_equal(b[i], c[i]):
                break
        #print(i, 'negative', f.path)
        a[i, :, :, :] = img
        b[i] = c[i]

        count -= 1

    # Save data for preview
    i = 0
    for face in train_data:
        ep_dir = os.path.join(train_dir, 'ep0000')
        label_dir = str(train_labels[i]).zfill(3)
        write_dir = os.path.join(ep_dir, label_dir)
        DataUtilities.prepare_dir(write_dir)
        cv2.imwrite(os.path.join(write_dir, str(i).zfill(4)+'.jpg'), (face*255).astype(dtype=np.uint8))
        i += 1
    i = 0
    for face in val_data:
        label_dir = str(val_labels[i]).zfill(3)
        write_dir = os.path.join(val_dir, label_dir)
        DataUtilities.prepare_dir(write_dir)
        cv2.imwrite(os.path.join(write_dir, str(i).zfill(4)+'.jpg'), (face*255).astype(dtype=np.uint8))
        i += 1

    #print(train_data.shape)
    #print(train_labels.shape)
    #print(val_data.shape)
    #print(val_labels.shape)
    return (train_data, train_labels, val_data, val_labels)

def train(args):
    log_dir = '../models/cascade'
    """if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)"""

    sess = tf.InteractiveSession()
 
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, np.prod(shape_image)], name='train_data')
        y_ = tf.placeholder(tf.float32, [None, n_class], name='labels')
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, (-1,)+shape_image)
        tf.summary.image('input', image_shaped_input, n_class)
        print(image_shaped_input)

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
        variable_summaries(weights)
        variable_summaries(biases)
        conv = tf.nn.conv2d(input, weights,
            strides = [1, 1, 1, 1], padding = 'SAME')
        return tf.nn.relu(conv + biases)

    def pool(input, size):
        return tf.nn.max_pool(
            input, 
            ksize = [1, size, size, 1], 
            strides = [1, size, size, 1], 
            padding = 'SAME'
        )

    # Convolutions
    with tf.variable_scope('conv1'):
        conv1 = conv_relu(image_shaped_input, kernel_size=3, depth=16)
        pool1 = pool(conv1, size = 2)

    shape = pool1.get_shape().as_list()
    flatten = tf.reshape(pool1, [-1, shape[1] * shape[2] * shape[3]])

    with tf.variable_scope('fc1'):
        fc1 = fully_connected_relu(flatten, size=16)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(fc1, keep_prob)

    with tf.variable_scope('out'):
        y = fully_connected(dropped, size=n_class)

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
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.0005).minimize(
            cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir+'/test')
    tf.global_variables_initializer().run()

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries
    train_data, train_labels, val_data, val_labels = get_data()

    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            xs = train_data.reshape((-1,)+shape_flat)
            ys = train_labels
            k = 0.5
        else:
            xs = val_data.reshape((-1,)+shape_flat)
            ys = val_labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(20000):
        if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                    feed_dict=feed_dict(True),
                                    options=run_options,
                                    run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
            if i % 200 == 199:
                print('Get new data')
                print()
                train_data, train_labels, val_data, val_labels = get_data()
    train_writer.close()
    test_writer.close()

def run(args):
    if args.train_id:
        model_dir = '../models/' + args.model + '_' + args.train_id
    else:
        model_dir = '../models/' + args.model
    print(model_dir)
    val_dir = os.path.join(model_dir, 'val')
    DataUtilities.prepare_dir(model_dir, 'val', empty=True)
    train_dir = os.path.join(model_dir, 'train')
    DataUtilities.prepare_dir(model_dir, 'train', empty=True)

    # Hyperparameters
    prep_denoise = False
    prep_equalize = True

    learn_rate = 0.0005
    epochs = 40
    steps = 4000

    train_count = 60
    val_count = 2400
    total_count = train_count + val_count

    train_positive_count = train_count//2
    train_negative_count = train_count - train_positive_count
    val_positive_count = val_count//3
    val_negative_count = val_count - val_positive_count
    positive_count = train_positive_count + val_positive_count
    negative_count = train_negative_count + val_negative_count

    a = np.zeros(shape=[total_count, 96, 64, 3], dtype=np.float32) # Data
    b = np.full(shape=[total_count, 1], fill_value=-1, dtype=np.int) # Labels
    c = np.zeros(shape=[total_count, 1], dtype=np.int) # Label assignment
    c[0:train_positive_count, :] = 1
    c[train_count:train_count+val_positive_count, :] = 1

    train_data = a[0:train_count, :, :, :]
    train_labels = b[0:train_count, :]
    val_data = a[train_count:total_count, :, :, :]
    val_labels = b[train_count:total_count, :]

    # Load positive data from dataset
    data_dir = '../data/face/sof/images'
    dset = SoF()
    dset.load_annotations('../data/face/sof/images/metadata.mat', 'sof')
    count = positive_count
    face_index = 0
    while count:
        anno = dset.get_face(face_index)
        face_index += 1
        if anno is None:
            face_index = 0
            continue
        #print(anno)
        filename = '_'.join((anno['file_prefix'], 'e0_nl_o'))
        filename = '.'.join((filename, 'jpg'))
        imgpath = os.path.normpath(os.path.join(data_dir, filename))

        img = cv2.imread(imgpath, 1)
        if img is None:
            continue
        src_shape = img.shape
        height, width, *rest = img.shape
        (x, y, w, h) = ImageUtilities.rect_fit_ar(anno['rect'].astype(dtype=np.int), [0, 0, width, height], 2/3, mrate=1.25)
        if w>0 and h>0:
            pass
        else:
            continue

        #face = img[y:y+h, x:x+w, :]
        face = ImageUtilities.transform_crop((x, y, w, h), img, r_intensity=1.0, p_intensity=0.)
        face = imresize(face, [96, 64])
        face = ImageUtilities.preprocess(face, convert_gray=None, equalize=prep_equalize, denoise=prep_denoise)
        face = np.array(face, dtype=np.float32)/255

        for i in range(len(b)):
            if c[i, 0]==1 and b[i, 0]!=c[i, 0]:
                break
        print(i, 'positive', imgpath)
        a[i, :, :, :] = face
        b[i, :] = c[i, 0]

        #'(\w{4})_(\d{5})_([mfMF])_(\d{2})(_([ioIO])_(fr|nf)_(cr|nc)_(no|hp|sd|sr)_(\d{4})_(\d)_(e0|en|em)_(nl|Gn|Gs|Ps)_([oemh]))*\.jpg'
        """
        canvas = ViewportManager().open('preview', shape=img.shape, blocks=(1, 2))
        ViewportManager().put('preview', img, (0, 0))
        ViewportManager().put('preview', face, (0, 1))
        ViewportManager().update('preview')

        k = ViewportManager().wait_key()
        if k in (ViewportManager.KEY_ENTER, ViewportManager.KEY_SPACE):
            pass
        """

        count -= 1

    # Load negative data
    count = negative_count
    while count:
        f = DirectoryWalker().get_a_file(directory='../data/face/processed/negative', filters=['.jpg'])
        image = cv2.imread(f.path, 1)
        image = ImageUtilities.preprocess(image, convert_gray=None, equalize=prep_equalize, denoise=prep_denoise)
        image = np.array(image, dtype=np.float32)/255

        for i in range(len(b)):
            if c[i, 0]==0 and b[i, 0]!=c[i, 0]:
                break
        print(i, 'negative', f.path)
        a[i, :, :, :] = image
        b[i, :] = c[i, 0]

        count -= 1

    # Validate with HOG_SVM face detection
    svm_val_correct = 0
    time_detect = 0.
    i = 0
    for face in val_data:
        gray = ImageUtilities.preprocess(face, convert_gray=cv2.COLOR_RGB2YCrCb, equalize=prep_equalize, denoise=prep_denoise)
        time_start = time.time()
        rects, landmarks = FaceDetector().detect(gray)
        time_diff = time.time() - time_start
        time_detect += time_diff
        if (len(rects) and val_labels[i]) or (len(rects)==0 and val_labels[i]==0):
            svm_val_correct += 1
        i += 1
    print(svm_val_correct, '/', i, ', acc:', svm_val_correct/i, ', time/detect:', time_detect/i)


    # Save data for preview
    i = 0
    for face in train_data:
        ep_dir = os.path.join(train_dir, 'ep0000')
        label_dir = str(train_labels[i]).zfill(3)
        write_dir = os.path.join(ep_dir, label_dir)
        DataUtilities.prepare_dir(write_dir)
        cv2.imwrite(os.path.join(write_dir, str(i).zfill(4)+'.jpg'), (face*255).astype(dtype=np.uint8))
        i += 1
    i = 0
    for face in val_data:
        label_dir = str(val_labels[i]).zfill(3)
        write_dir = os.path.join(val_dir, label_dir)
        DataUtilities.prepare_dir(write_dir)
        cv2.imwrite(os.path.join(write_dir, str(i).zfill(4)+'.jpg'), (face*255).astype(dtype=np.uint8))
        i += 1

    print('Train data ready')

    depth = 5
    out_weights = np.zeros((depth,), dtype=np.float)
    out_weights[0] = 1.

    model = SimpleClassifier(args.model, args.train_id, {'learn_rate': learn_rate})
    classifier = model.get_estimator(out_weights, model_dir)

    # Set up logging for predictions
    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=10)
    tf.logging.set_verbosity(tf.logging.INFO)

    for epoch in range(epochs):
        print('train_labels', train_labels.shape)
        print('val_labels', val_labels.shape)

        """
        onehot = np.zeros((len(train_labels), depth), dtype=np.float)
        onehot[np.arange(len(train_labels)), train_labels] = 1.
        train_labels = onehot

        onehot = np.zeros((len(val_labels), depth), dtype=np.float)
        onehot[np.arange(len(val_labels)), val_labels] = 1.
        val_labels = onehot"""

        print('train_data', train_data.shape)
        print('train_labels', train_labels.shape)
        print('val_data', val_data.shape)
        print('val_labels', val_labels.shape)

        try:
            start_time = time.time()
            # Evaluate the model and print results
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'x': val_data},
                y=val_labels,
                num_epochs=1,
                shuffle=False)
            eval_results = classifier.evaluate(input_fn=eval_input_fn)
            print(eval_results)
            duration = time.time() - start_time
            print('duration', duration, 'per sample', duration/len(val_data))

            
        except ValueError:
            pass

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': train_data},
            y=train_labels,
            batch_size=train_count,
            num_epochs=None,
            shuffle=True)
        classifier.train(
            input_fn=train_input_fn,
            steps=steps,
            hooks=[logging_hook])

        # Get new train set
        b[0:train_count] = -1
        c[0:train_positive_count, :] = 1
        c[train_positive_count:train_positive_count+train_negative_count, :] = 0
        # Load positive data from dataset
        count = train_positive_count
        while count:
            anno = dset.get_face(face_index)
            face_index += 1
            if anno is None:
                face_index = 0
                continue
            #print(anno)
            filename = '_'.join((anno['file_prefix'], 'e0_nl_o'))
            filename = '.'.join((filename, 'jpg'))
            imgpath = os.path.normpath(os.path.join(data_dir, filename))

            img = cv2.imread(imgpath, 1)
            if img is None:
                continue
            src_shape = img.shape
            height, width, *rest = img.shape
            (x, y, w, h) = ImageUtilities.rect_fit_ar(anno['rect'].astype(dtype=np.int), [0, 0, width, height], 2/3, mrate=1.25)
            if w>0 and h>0:
                pass
            else:
                continue

            #face = img[y:y+h, x:x+w, :]
            face = ImageUtilities.transform_crop((x, y, w, h), img, r_intensity=1.0, p_intensity=0.)
            face = imresize(face, [96, 64])
            face = ImageUtilities.preprocess(face, convert_gray=None, equalize=prep_equalize, denoise=prep_denoise)
            face = np.array(face, dtype=np.float32)/255

            for i in range(len(b)):
                if c[i, 0]==1 and b[i, 0]!=c[i, 0]:
                    break
            print(i, 'positive', imgpath)
            a[i, :, :, :] = face
            b[i, :] = c[i, 0]

            count -= 1

        # Load negative data
        count = train_negative_count
        while count:
            f = DirectoryWalker().get_a_file(directory='../data/face/processed/negative', filters=['.jpg'])
            image = cv2.imread(f.path, 1)
            image = ImageUtilities.preprocess(image, convert_gray=None, equalize=prep_equalize, denoise=prep_denoise)
            image = np.array(image, dtype=np.float32)/255

            for i in range(len(b)):
                if c[i, 0]==0 and b[i, 0]!=c[i, 0]:
                    break
            print(i, 'negative', f.path)
            a[i, :, :, :] = image
            b[i, :] = c[i, 0]

            count -= 1

        # Save data for preview
        i = 0
        for face in train_data:
            ep_dir = os.path.join(train_dir, 'ep'+str(epoch+1).zfill(4))
            label_dir = str(train_labels[i]).zfill(3)
            write_dir = os.path.join(ep_dir, label_dir)
            DataUtilities.prepare_dir(write_dir)
            cv2.imwrite(os.path.join(write_dir, str(i).zfill(4)+'.jpg'), (face*255).astype(dtype=np.uint8))
            i += 1

    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(layers)