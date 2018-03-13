from facer.train import *
from facer.datagen import *
from facer.predict import *

def main():
    if ARGS.test=='train':
        train(ARGS)
    elif ARGS.test=='gen':
        gen(ARGS)
    elif ARGS.test=='predict':
        predict(ARGS)

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
        '--source_dir',
        type=str,
        default='../data/face/wiki-face/extracted/wiki',
        help='Path to the data.'
    )
    parser.add_argument(
        '--annotations',
        type=str,
        default='../data/face/wiki-face/extracted/wiki/wiki.mat',
        help='Path to annotations.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='afanet11',
    )
    parser.add_argument(
        '--subset',
        type=str,
        default='positive',
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