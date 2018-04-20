import logging, sys

LOG_DIRECTORY = '../log/'
formatter = logging.Formatter('[%(asctime)s] %(levelno)s (%(process)d) %(module)s: %(message)s')

sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)

fh = logging.FileHandler(LOG_DIRECTORY + 'arobcv.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

logger = logging.getLogger('ALogger')
logger.propagate = False
logger.setLevel(logging.DEBUG)
logger.addHandler(sh)
logger.addHandler(fh)

def my_handler(type, value, tb):
    logger.exception('Uncaught exception: {0}'.format(str(value)))

# Install exception handler
sys.excepthook = my_handler

debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical