from facer.train import *
from facer.datagen import *
from facer.predict import *
from server.server import *
from emotion.emotion_recognition import fxpress, fxpress_train
from facer.emotion import EmotionClassifier
from facer.face_app import FaceApplications

def main():
    if ARGS.test=='train':
        train(ARGS)
    elif ARGS.test=='gen':
        gen(ARGS)
    elif ARGS.test=='predict':
        predict(ARGS)
    elif ARGS.test=='server':
        server_start(ARGS)
    elif ARGS.test=='server_production':
        server_start(ARGS, ARGS.port)
    elif ARGS.test=='hnm':
        hnm(ARGS)
    elif ARGS.test=='val':
        val(ARGS)
    elif ARGS.test=='fer':
        fer(ARGS)
    elif ARGS.test=='fxpress':
        fxpress(ARGS)
    elif ARGS.test=='fxpress_train':
        fxpress_train(ARGS)
    elif ARGS.test=='emoc':
        classifier = EmotionClassifier()
        classifier.build_network(ARGS)
        classifier.val(ARGS)
    elif ARGS.test=='face_app':
        face_app = FaceApplications()
        face_app.detect()

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Test units for face applications""")
    parser.add_argument(
        '--test',
        type=str,
        default='',
        help='Name of the test unit to run.'
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Display window and wait for user input, for preview.'
    )
    parser.add_argument(
        '--nolog',
        action='store_true',
        help='Skip logging to save disk space during training.'
    )
    parser.add_argument(
        '--subset',
        type=str,
        default='',
        help='Name of subset of dataset to be processed.'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=0,
        help='Count of entries to be processed or generated.'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port number of web server to listen to.'
    )
    parser.add_argument(
        '--model',
        type=str,
        #default='./server/models/12-net/model.ckpt',
        default='../models/cascade/checkpoint/model.ckpt',
        help='Path and prefix of Tensorflow checkpoint.'
    )
    parser.add_argument(
        '--cascade',
        type=int,
        default=12,
        help='Level of cascade CNN '
    )
    parser.add_argument(
        '--annotations',
        type=str,
        default='../data/face/wiki-face/extracted/wiki/wiki.mat',
        help='Path to annotations.'
    )
    parser.add_argument(
        '--train_id',
        type=str,
        default='',
    )
    ARGS, unknown = parser.parse_known_args()
    if unknown:
        print(unknown)
        raise Exception('Unknown argument')
    main()