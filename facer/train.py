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

def run(args):
    print('do something...')

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