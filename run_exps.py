import os
import random
import string
import subprocess
from datetime import datetime


def create_exp_dirs(root_dir, task):
    if not os.path.exists(root_dir):
        raise FileNotFoundError(
            f"Root directory '{root_dir}', where the directory of the experiment will be created, must exist")

    formatted_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(root_dir, task)
    rand_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=3))
    output_dir = output_dir + "_" + formatted_timestamp + "_" + rand_suffix
    exp_args_output_dir = os.path.join(output_dir, "output")
    exp_args_logging_dir = os.path.join(output_dir, "logging")

    os.makedirs(exp_args_output_dir, exist_ok=True)
    os.makedirs(exp_args_logging_dir, exist_ok=True)

    return exp_args_output_dir, exp_args_logging_dir


class MainRunnerArgs:
    def __init__(self, root_dir, scratch, dataset_name, data_path, excel_path, bert_model, modality, batch_size,
                 pre_epoch, finetune_epoch, patience):
        self.root_dir = root_dir
        self.scratch = scratch
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.excel_path = excel_path
        self.bert_model = bert_model
        self.modality = modality
        self.batch_size = batch_size
        self.pre_epoch = pre_epoch
        self.finetune_epoch = finetune_epoch
        self.patience = patience


def run_single_exp(main_runner_args):
    if main_runner_args.scratch == "yes":
        task = "fine_tune_from_scratch"
        exp_dirs = create_exp_dirs(main_runner_args.root_dir, task)
        if exp_dirs is None:
            return
        output_dir, logging_dir = exp_dirs
        subprocess.run([
            "python", "main_runner.py",
            "--task", task,
            "--bert_model_name", main_runner_args.bert_model,
            "--use_modality", main_runner_args.modality,
            "--per_device_train_batch_size", main_runner_args.batch_size,
            "--num_train_epochs", main_runner_args.finetune_epoch,
            "--patience", main_runner_args.patience,
            "--dataset_name", main_runner_args.dataset_name,
            "--data_path", main_runner_args.data_path,
            "--output_dir", output_dir,
            "--logging_dir", logging_dir,
            "--save_excel_path", main_runner_args.excel_path
        ])
    else:
        task = "pretrain"
        pretrain_exp_dirs = create_exp_dirs(main_runner_args.root_dir, task)
        if pretrain_exp_dirs is None:
            return
        pretrain_output_dir, pretrain_logging_dir = pretrain_exp_dirs
        subprocess.run([
            "python", "main_runner.py",
            "--task", task,
            "--bert_model_name", main_runner_args.bert_model,
            "--use_modality", main_runner_args.modality,
            "--per_device_train_batch_size", main_runner_args.batch_size,
            "--num_train_epochs", main_runner_args.pre_epoch,
            "--patience", main_runner_args.patience,
            "--dataset_name", main_runner_args.dataset_name,
            "--data_path", main_runner_args.data_path,
            "--output_dir", pretrain_output_dir,
            "--logging_dir", pretrain_logging_dir,
            "--save_excel_path", main_runner_args.excel_path
        ])

        task = "fine_tune"
        fine_tune_exp_dirs = create_exp_dirs(main_runner_args.root_dir, task)
        if fine_tune_exp_dirs is None:
            return
        fine_tune_output_dir, fine_tune_logging_dir = fine_tune_exp_dirs
        subprocess.run([
            "python", "main_runner.py",
            "--task", task,
            "--bert_model_name", main_runner_args.bert_model,
            "--use_modality", main_runner_args.modality,
            "--per_device_train_batch_size", main_runner_args.batch_size,
            "--num_train_epochs", main_runner_args.finetune_epoch,
            "--patience", main_runner_args.patience,
            "--dataset_name", main_runner_args.dataset_name,
            "--data_path", main_runner_args.data_path,
            "--output_dir", fine_tune_output_dir,
            "--logging_dir", fine_tune_logging_dir,
            "--pretrained_model_dir", pretrain_output_dir,
            "--save_excel_path", main_runner_args.excel_path
        ])


def generate_dataset(dataset_name, sent_num, kw_num):
    subprocess.run([
        "python", "crmm/data_acquisition/5sec_text_sumy.py",
        "--sentences_num", sent_num,
        "--dataset_name", dataset_name
    ])
    subprocess.run([
        "python", "crmm/data_acquisition/6sec_keywords_extraction.py",
        "--prev_step_sentences_num", sent_num,
        "--keywords_num", kw_num,
        "--dataset_name", dataset_name
    ])


def split_dataset(data_config):
    subprocess.run([
        "python", "crmm/data_acquisition/7.2dataset_split.py",
        "--dataset_name", data_config["dataset_name"],
        "--prev_step_sentences_num", data_config["sent_num"],
        "--prev_step_keywords_num", data_config["kw_num"],
        "--n_class", data_config["n_class"],
        "--split_method", data_config["split_method"],
        "--train_years", data_config["train_years"],
        "--test_years", data_config["test_years"],
    ])


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@ EXPS @@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def run_pre_epoch_exps(data_config):
    main_runner_args = MainRunnerArgs(
        root_dir="./exps",
        scratch="no",
        dataset_name=data_config["dataset_name"],
        data_path=data_config["data_path"],
        excel_path=f"./excel/{data_config['dataset_name']}_cls{data_config['n_class']}_pre_epoch.xlsx",
        bert_model="prajjwal1/bert-tiny",
        modality="num,cat,text",
        batch_size="300",
        pre_epoch="5",
        finetune_epoch="200",
        patience="1000"
    )
    for pre_epoch in [1]:
        # for pre_epoch in [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]:
        main_runner_args.pre_epoch = str(pre_epoch)
        run_single_exp(main_runner_args)

    # main_runner_args.scratch = "yes"
    # run_single_exp(main_runner_args)


def run_modality_exps(data_config):
    main_runner_args = MainRunnerArgs(
        root_dir="./exps",
        scratch="no",
        dataset_name=data_config["dataset_name"],
        data_path=data_config["data_path"],
        excel_path=f"./excel/{data_config['dataset_name']}_cls{data_config['n_class']}_modality.xlsx",
        bert_model="prajjwal1/bert-tiny",
        modality="num,cat,text",
        batch_size="300",
        pre_epoch="5",
        finetune_epoch="200",
        patience="1000"
    )
    for modality in ["num", "cat", "text", "num,cat", "num,text", "cat,text", "num,cat,text"]:
        main_runner_args.modality = modality
        run_single_exp(main_runner_args)


if __name__ == "__main__":
    data_config = {
        "dataset_name": "cr2",  # !
        "sent_num": "10",
        "kw_num": "20",
        "n_class": "6",  # !
        "split_method": "mixed",
        "train_years": "-1",
        "test_years": "-1",
    }
    data_path = f"./data/{data_config['dataset_name']}_cls{data_config['n_class']}_" \
                f"{data_config['split_method']}_" \
                f"st{data_config['sent_num']}_kw{data_config['kw_num']}"
    data_config["data_path"] = data_path

    # split_dataset(data_config)

    # run_pre_epoch_exps(data_config)

    run_modality_exps(data_config)
