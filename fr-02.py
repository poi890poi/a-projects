from facer.train import *

def main():
    print('do nothing...')
    run(ARGS)

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Clean up images and transform to generate more samples""")
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
        default='afanet',
    )
    ARGS, unknown = parser.parse_known_args()
    if unknown:
        print(unknown)
        raise Exception('Unknown argument')
    main()