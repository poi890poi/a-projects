from shared.server import *

def main():
    print('do nothing...')
    run(ARGS)

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="""\
        Server for testing""")
    ARGS, unknown = parser.parse_known_args()
    if unknown:
        print(unknown)
        raise Exception('Unknown argument')
    main()