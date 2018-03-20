from facer.train import *
from facer.datagen import *
from facer.predict import *
from server.server import *

def main():
    if ARGS.test=='train':
        train(ARGS)
    elif ARGS.test=='gen':
        gen(ARGS)
    elif ARGS.test=='predict':
        predict(ARGS)
    elif ARGS.test=='server':
        server_start(ARGS)
    elif ARGS.test=='hnm':
        hnm(ARGS)
    elif ARGS.test=='val':
        val(ARGS)

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
        '--model',
        type=str,
        #default='./server/models/12-net/model.ckpt',
        default='../models/cascade/checkpoint/model.ckpt',
        help='Path and prefix of Tensorflow checkpoint.'
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