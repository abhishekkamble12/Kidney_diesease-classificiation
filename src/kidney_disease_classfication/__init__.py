# IMPORT OPERATION 
import os 
import sys 
import logging


logging_str = "[%(asctime)s] - [%(levelname)s] - [%(lineno)d] - [%(name)s] - [%(message)s]"

log_dir = "logging"
log_filepath = os.path.join(log_dir, "running_logs.log")
# make the directory 
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

# creating the object 
logger = logging.getLogger("kidney_disease_classfication")
