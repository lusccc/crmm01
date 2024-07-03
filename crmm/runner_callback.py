from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.utils import logging
import time
logger = logging.get_logger('transformers')


class TrainerLoggerCallback(TrainerCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):
        # A bare [`TrainerCallback`] that just prints the logs.
        _ = logs.pop("total_flos", None)
        # if state.is_local_process_zero:
        #     logger.info(logs)


class CrmmTrainerCallback(TrainerCallback):

    def __init__(self, runner) -> None:
        self.runner = runner

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_epoch_end(args, state, control, **kwargs)
        self.runner.predict_on_test()


class TimeCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start_time = None
        self.eval_start_time = None
        self.epoch_times = []
        self.eval_times = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        logger.info(f"Epoch {state.epoch} took {epoch_time:.2f} seconds")

    def on_evaluate(self, args, state, control, **kwargs):
        if self.eval_start_time:
            eval_time = time.time() - self.eval_start_time
            self.eval_times.append(eval_time)
            logger.info(f"Evaluation took {eval_time:.2f} seconds")

    def on_evaluate_begin(self, args, state, control, **kwargs):
        self.eval_start_time = time.time()
