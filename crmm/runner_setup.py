import json
import logging as syslogging
import os
import random
import string
from datetime import datetime

import torch
from transformers.utils import logging

from .arguments import CrmmTrainingArguments
from .utils import utils
from .utils.log_handler import ColorFormatter

logger = logging.get_logger('transformers')


def setup(exp_args: CrmmTrainingArguments):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """
    torch.set_num_threads(12)
    initial_timestamp = datetime.now()
    root_dir = exp_args.root_dir
    if not os.path.isdir(root_dir):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(root_dir))

    output_dir = os.path.join(root_dir, exp_args.task)

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
    output_dir += "_" + formatted_timestamp + "_" + rand_suffix
    exp_args.output_dir = os.path.join(output_dir, 'output')
    exp_args.logging_dir = os.path.join(output_dir, 'logging')
    utils.create_dirs([exp_args.output_dir, exp_args.logging_dir])

    # logzero.logfile(os.path.join(root_dir, 'output.log'), backupCount=3)
    # logger.addHandler(LogFormatHandler())
    # syslogging.basicConfig(filename=os.path.join(exp_args.output_dir, 'output.log'),
    #                        filemode='a',
    #                        format='%(asctime)s %(filename)-18s %(levelname)-8s: %(message)s',
    #                        level=logging.DEBUG,
    #                        force=True)

    color_fmt = ColorFormatter()
    file_handler = syslogging.FileHandler(os.path.join(exp_args.output_dir, 'output.log'), )
    file_handler.setFormatter(color_fmt)

    console_handler = syslogging.StreamHandler()
    console_handler.setFormatter(color_fmt)

    # !!! note:  disable_default_handler!
    logging.disable_default_handler()
    logging.set_verbosity_debug()
    logging.add_handler(file_handler)
    logging.add_handler(console_handler)

    # setup dbn loggers
    dbn_loggers = [
        logging.get_logger('learnergy.models.deep.dbn'),
        logging.get_logger('learnergy.models.bernoulli.rbm'),
        logging.get_logger('learnergy.models.extra.sigmoid_rbm'),
        logging.get_logger('learnergy.core.model'),
        logging.get_logger('learnergy.core.dataset'),
    ]
    for lg in dbn_loggers:
        lg.handlers.clear()
        lg.addHandler(file_handler)
        lg.addHandler(console_handler)


    # Save configuration as a (pretty) json file
    with open(os.path.join(exp_args.output_dir, 'training_arguments.json'), 'w') as f:
        json.dump(exp_args.to_sanitized_dict(), f, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(exp_args.output_dir))

    return exp_args
