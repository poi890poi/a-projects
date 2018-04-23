import logging, sys, traceback

LOG_DIRECTORY = '../log/'
formatter = logging.Formatter('[%(asctime)s] %(levelname)s (%(process)d) %(module)s: %(message)s')

sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)

fh = logging.handlers.RotatingFileHandler(LOG_DIRECTORY + 'rbtimg.log', maxBytes=16*1024*1024, backupCount=64)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

logger = logging.getLogger('ALogger')
logger.propagate = False
logger.setLevel(logging.DEBUG)
logger.addHandler(sh)
logger.addHandler(fh)

def my_handler(type, value, tb):
    #logger.exception('Uncaught exception: {0}'.format(str(value)))
    logger.error('\n'.join(['Uncaught Exception'] + list(traceback.format_tb(tb, limit=32))))

# Install exception handler
sys.excepthook = my_handler

debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical