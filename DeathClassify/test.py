import logging
import datetime
import sys

logger = logging.getLogger('mylogger')
logger.setLevel(logging.INFO)
rf_handler = logging.StreamHandler(sys.stderr)  # 默认是sys.stderr
rf_handler.setLevel(logging.DEBUG)
# rf_handler = logging.handlers.TimedRotatingFileHandler('all.log', when='midnight', interval=1, backupCount=7, atTime=datetime.time(0, 0, 0, 0))
rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))

f_handler = logging.FileHandler('training.log')
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))

logger.addHandler(rf_handler)
logger.addHandler(f_handler)

# logger.debug('debug message')
# logger.info('info message')
# logger.warning('warning message')
# logger.error('error message')
# logger.critical('critical message')
import config
import csv
with open(config.train_process_20x_detail_data_savefile, 'w',
          newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(['1', '2'])