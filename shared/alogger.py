import logging, sys, traceback

LOG_DIRECTORY = '../log/'
formatter = logging.Formatter('[%(asctime)s] %(levelname)s (%(process)d) %(module)s Ln%(lineno)d: %(message)s')

sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)

fh = logging.handlers.RotatingFileHandler(LOG_DIRECTORY + 'rbtimg.log', maxBytes=32*1024*1024, backupCount=16)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

logger = logging.getLogger('ALogger')
logger.propagate = False
logger.setLevel(logging.DEBUG)
logger.addHandler(sh)
logger.addHandler(fh)

def my_handler(type, value, tb):
    logger.error('\n'.join(['Uncaught Exception'] + list(traceback.format_tb(tb, limit=32)) + [type.__name__+': '+str(value),]))

# Install exception handler
sys.excepthook = my_handler

debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical