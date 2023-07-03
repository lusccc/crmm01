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
    def __init__(self, root_dir, scratch, dataset_name, dataset_info, data_path, excel_path, bert_model,
                 use_hf_pretrained_bert_in_pretrain, freeze_bert_params, modality, modality_fusion_method,
                 text_cols, batch_size, pre_epoch, finetune_epoch, patience):
        self.root_dir = root_dir
        self.scratch = scratch
        self.dataset_name = dataset_name
        self.dataset_info = dataset_info
        self.data_path = data_path
        self.excel_path = excel_path
        self.bert_model = bert_model
        self.use_hf_pretrained_bert_in_pretrain = use_hf_pretrained_bert_in_pretrain
        self.freeze_bert_params = freeze_bert_params
        self.modality = modality
        self.modality_fusion_method = modality_fusion_method
        self.text_cols = text_cols
        self.batch_size = batch_size
        self.pre_epoch = pre_epoch
        self.finetune_epoch = finetune_epoch
        self.patience = patience


def run_single_exp(main_runner_args):
    if main_runner_args.freeze_bert_params == "default_best":
        freeze_bert_params_in_pretrain = "true"
        freeze_bert_params_in_finetune = "false"
    else:
        freeze_bert_params_in_pretrain = main_runner_args.freeze_bert_params
        freeze_bert_params_in_finetune = main_runner_args.freeze_bert_params

    if main_runner_args.scratch == "yes":
        task = "fine_tune_from_scratch"
        exp_dirs = create_exp_dirs(main_runner_args.root_dir, task)
        if exp_dirs is None:
            return
        output_dir, logging_dir = exp_dirs
        cmd = [
            "python", "main_runner.py",
            "--task", task,
            "--bert_model_name", main_runner_args.bert_model,
            "--use_hf_pretrained_bert_in_pretrain", main_runner_args.use_hf_pretrained_bert_in_pretrain,
            "--freeze_bert_params", freeze_bert_params_in_finetune,
            "--use_modality", main_runner_args.modality,
            "--modality_fusion_method", main_runner_args.modality_fusion_method,
            "--text_cols", main_runner_args.text_cols,
            "--per_device_train_batch_size", main_runner_args.batch_size,
            "--num_train_epochs", main_runner_args.finetune_epoch,
            "--patience", main_runner_args.patience,
            "--dataset_name", main_runner_args.dataset_name,
            "--dataset_info", main_runner_args.dataset_info,
            "--data_path", main_runner_args.data_path,
            "--output_dir", output_dir,
            "--logging_dir", logging_dir,
            "--save_excel_path", main_runner_args.excel_path
        ]
        subprocess.run(cmd)
    else:
        task = "pretrain"
        pretrain_exp_dirs = create_exp_dirs(main_runner_args.root_dir, task)
        if pretrain_exp_dirs is None:
            return
        pretrain_output_dir, pretrain_logging_dir = pretrain_exp_dirs
        pretrain_cmd = [
            "python", "main_runner.py",
            "--task", task,
            "--bert_model_name", main_runner_args.bert_model,
            "--use_hf_pretrained_bert_in_pretrain", main_runner_args.use_hf_pretrained_bert_in_pretrain,
            "--freeze_bert_params", freeze_bert_params_in_pretrain,
            "--use_modality", main_runner_args.modality,
            "--modality_fusion_method", main_runner_args.modality_fusion_method,
            "--text_cols", main_runner_args.text_cols,
            "--per_device_train_batch_size", main_runner_args.batch_size,
            "--num_train_epochs", main_runner_args.pre_epoch,
            "--patience", main_runner_args.patience,
            "--dataset_name", main_runner_args.dataset_name,
            "--dataset_info", main_runner_args.dataset_info,
            "--data_path", main_runner_args.data_path,
            "--output_dir", pretrain_output_dir,
            "--logging_dir", pretrain_logging_dir,
            "--save_excel_path", main_runner_args.excel_path
        ]
        subprocess.run(pretrain_cmd)
        task = "fine_tune"
        fine_tune_exp_dirs = create_exp_dirs(main_runner_args.root_dir, task)
        if fine_tune_exp_dirs is None:
            return
        fine_tune_output_dir, fine_tune_logging_dir = fine_tune_exp_dirs
        cmd = [
            "python", "main_runner.py",
            "--task", task,
            "--bert_model_name", main_runner_args.bert_model,
            "--use_hf_pretrained_bert_in_pretrain", main_runner_args.use_hf_pretrained_bert_in_pretrain,
            "--freeze_bert_params", freeze_bert_params_in_finetune,
            "--use_modality", main_runner_args.modality,
            "--modality_fusion_method", main_runner_args.modality_fusion_method,
            "--text_cols", main_runner_args.text_cols,
            "--per_device_train_batch_size", main_runner_args.batch_size,
            "--num_train_epochs", main_runner_args.finetune_epoch,
            "--patience", main_runner_args.patience,
            "--dataset_name", main_runner_args.dataset_name,
            "--dataset_info", main_runner_args.dataset_info,
            "--data_path", main_runner_args.data_path,
            "--output_dir", fine_tune_output_dir,
            "--logging_dir", fine_tune_logging_dir,
            "--pretrained_model_dir", pretrain_output_dir,
            "--save_excel_path", main_runner_args.excel_path
        ]
        subprocess.run(cmd)



def generate_dataset(data_config):
    dataset_name = data_config["dataset_name"]
    sent_num = data_config["sent_num"]
    kw_num = data_config["kw_num"]
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
    extra_info = "freeze_bert_in_pre"
    main_runner_args = MainRunnerArgs(
        root_dir="./exps",
        scratch="no",
        dataset_name=data_config["dataset_name"],
        dataset_info="",
        data_path=data_config["data_path"],
        excel_path=f"./excel/{data_config['dataset_name']}_cls{data_config['n_class']}_({extra_info}).xlsx",  # !
        bert_model="prajjwal1/bert-tiny",
        use_hf_pretrained_bert_in_pretrain="true",
        freeze_bert_params="default_best",
        modality="num,cat,text",
        modality_fusion_method="conv",
        text_cols="secKeywords",
        batch_size="300",
        pre_epoch="5",
        finetune_epoch="200",
        patience="1000"
    )
    for pre_epoch in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
        # for pre_epoch in [15, 20, 30]:
        # for pre_epoch in [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]:
        main_runner_args.pre_epoch = str(pre_epoch)
        run_single_exp(main_runner_args)

    main_runner_args.scratch = "yes"
    run_single_exp(main_runner_args)


def run_conv_fusion_exp(data_config, pre_epoch):
    extra_info = "conv_or_concat"
    main_runner_args = MainRunnerArgs(
        root_dir="./exps",
        scratch="no",
        dataset_name=data_config["dataset_name"],
        dataset_info="",
        data_path=data_config["data_path"],
        excel_path=f"./excel/{data_config['dataset_name']}_cls{data_config['n_class']}_({extra_info}).xlsx",  # !
        bert_model="prajjwal1/bert-tiny",
        use_hf_pretrained_bert_in_pretrain="true",
        freeze_bert_params="default_best",
        modality="num,cat,text",
        modality_fusion_method="concat",  # !
        text_cols="secKeywords",
        batch_size="300",
        pre_epoch=pre_epoch,
        finetune_epoch="200",
        patience="1000"
    )
    run_single_exp(main_runner_args)


def run_kw_or_txt_exp(data_config, pre_epoch):
    extra_info = "kw_or_txt"

    text_cols = "secKeywords"
    main_runner_args = MainRunnerArgs(
        root_dir="./exps",
        scratch="no",
        dataset_name=data_config["dataset_name"],
        dataset_info="kw",
        data_path=data_config["data_path"],
        excel_path=f"./excel/{data_config['dataset_name']}_cls{data_config['n_class']}_({extra_info}).xlsx",  # !
        bert_model="prajjwal1/bert-tiny",
        use_hf_pretrained_bert_in_pretrain="true",
        freeze_bert_params="default_best",
        modality="num,cat,text",
        modality_fusion_method="conv",
        text_cols=text_cols,
        batch_size="300",
        pre_epoch=pre_epoch,
        finetune_epoch="200",
        patience="1000"
    )
    run_single_exp(main_runner_args)

    main_runner_args.text_cols = "secText"
    main_runner_args.dataset_info = "sec_txt"
    run_single_exp(main_runner_args)


def run_modality_exps(data_config, pre_epoch):
    scratch = "no"
    # scratch = "yes"

    bert_model = "prajjwal1/bert-tiny"
    # bert_model = "bert-base-uncased"

    use_hf_pretrained_bert_in_pretrain = "true"
    # use_hf_pretrained_bert_in_pretrain = "false"

    freeze_bert_params = "default_best"

    text_cols = "secKeywords"
    # text_cols = "secText"

    batch_size = "300"
    # batch_size = "100"

    modalities = ["num", "cat", "text", "num,cat", "num,text", "cat,text", "num,cat,text"]
    # modalities = ["text", "num,text", "cat,text", "num,cat,text"]
    # modalities = ["num", "cat", "text", "num,cat", "num,text", "cat,text",]
    # modalities = [ "text",  "num,text", "cat,text", "num,cat,text"]
    # modalities = ["num,cat,text"]
    # modalities = ["num,cat"]

    # extra_info = 'None'
    # extra_info = 'secItem2'
    extra_info = '0621_aft_pre10'

    main_runner_args = MainRunnerArgs(
        root_dir="./exps",
        scratch=scratch,
        dataset_name=data_config["dataset_name"],
        dataset_info="",
        data_path=data_config["data_path"],
        excel_path=f"./excel/{data_config['dataset_name']}_cls{data_config['n_class']}"
                   f"_modality_txtcol({text_cols})_({extra_info}).xlsx",
        bert_model=bert_model,
        use_hf_pretrained_bert_in_pretrain=use_hf_pretrained_bert_in_pretrain,
        freeze_bert_params=freeze_bert_params,
        modality="num,cat,text",  # ignore
        modality_fusion_method="conv",
        text_cols=text_cols,
        batch_size=batch_size,  # !
        pre_epoch=pre_epoch,  # !
        finetune_epoch="200",
        patience="1000"
    )
    for modality in modalities:
        main_runner_args.modality = modality
        run_single_exp(main_runner_args)


def run_rolling_window_exps(data_config, pre_epoch):
    data_config['split_method'] = 'by_year'

    # ------ EXPAND WHILE ROLLING FORWARD -------
    # train_years = [
    #     [2010, 2011, 2012],
    #     [2010, 2011, 2012, 2013],
    #     [2010, 2011, 2012, 2013, 2014],
    #     [2010, 2011, 2012, 2013, 2014, 2015],
    # ]
    # train_years = [
    #     [2011, 2012],
    #     [2012, 2013],
    #     [2013, 2014],
    #     [2014, 2015],
    # ]
    # train_years = [
    #     [2010, 2011, 2012],
    #     [2011, 2012, 2013],
    #     [2012, 2013, 2014],
    #     [2013, 2014, 2015],
    # ]
    # test_years = [
    #     2013,
    #     2014,
    #     2015,
    #     2016
    # ]

    # ----- FIXED WINDOW, MULTIPLE AHEAD
    train_years = [
        [2010, 2011, 2012],
        [2010, 2011, 2012],
        [2010, 2011, 2012],
        [2010, 2011, 2012],
        # ---
        [2011, 2012, 2013],
        [2011, 2012, 2013],
        [2011, 2012, 2013],
        # ---
        [2012, 2013, 2014],
        [2012, 2013, 2014],
        # ---
        [2013, 2014, 2015],
    ]
    test_years = [
        2013,
        2014,
        2015,
        2016,
        # ---
        2014,
        2015,
        2016,
        # ---
        2015,
        2016,
        # ---
        2016,
    ]
    for train_year, test_year in zip(train_years, test_years):
        train_years_str = ','.join(map(str, train_year))
        test_years_str = str(test_year)
        data_config['train_years'] = train_years_str
        data_config['test_years'] = test_years_str
        print(f'run_rolling_window_exps: {data_config}')
        split_dataset(data_config)
        data_config["data_path"] = f"./data/{data_config['dataset_name']}_cls{data_config['n_class']}_" \
                                   f"{data_config['split_method']}_" \
                                   f"st{data_config['sent_num']}_kw{data_config['kw_num']}_" \
                                   f'{",".join(map(str, train_year))}_{test_year}'

        extra_info = 'rolling_window_fixed_test_many_0627(findpre_all)'
        main_runner_args = MainRunnerArgs(
            root_dir="./exps",
            scratch='no',
            dataset_name=data_config["dataset_name"],
            dataset_info=f'{",".join(map(str, train_year))}_{test_year}',
            data_path=data_config["data_path"],
            excel_path=f"./excel/{data_config['dataset_name']}_cls{data_config['n_class']}"
                       f"_rolling_window_({extra_info}).xlsx",
            bert_model="prajjwal1/bert-tiny",
            use_hf_pretrained_bert_in_pretrain="true",
            freeze_bert_params="default_best",
            modality="num,cat,text",
            modality_fusion_method="conv",
            text_cols="secKeywords",
            batch_size="300",  # !
            pre_epoch=pre_epoch,  # !
            finetune_epoch="200",
            patience="1000"
        )
        run_single_exp(main_runner_args)


def run_benchmark(data_config, extra='0701'):
    subprocess.run([
        "python", "crmm/benchmark_model_comparison.py",
        "--data_path", data_config["data_path"],
        "--excel_path", f"./excel/benchmark_{data_config['dataset_name']}_cls{data_config['n_class']}_{extra}.xlsx",
    ])


def run_benchmark_rolling_window(data_config):
    data_config['split_method'] = 'by_year'
    train_years = [
        [2010, 2011, 2012],
        [2010, 2011, 2012],
        [2010, 2011, 2012],
        [2010, 2011, 2012],
        # ---
        [2011, 2012, 2013],
        [2011, 2012, 2013],
        [2011, 2012, 2013],
        # ---
        [2012, 2013, 2014],
        [2012, 2013, 2014],
        # ---
        [2013, 2014, 2015],
    ]
    test_years = [
        2013,
        2014,
        2015,
        2016,
        # ---
        2014,
        2015,
        2016,
        # ---
        2015,
        2016,
        # ---
        2016,
    ]
    for train_year, test_year in zip(train_years, test_years):
        train_years_str = ','.join(map(str, train_year))
        test_years_str = str(test_year)
        data_config['train_years'] = train_years_str
        data_config['test_years'] = test_years_str
        print(f'run_rolling_window_exps: {data_config}')
        split_dataset(data_config)
        data_config["data_path"] = f"./data/{data_config['dataset_name']}_cls{data_config['n_class']}_" \
                                   f"{data_config['split_method']}_" \
                                   f"st{data_config['sent_num']}_kw{data_config['kw_num']}_" \
                                   f'{",".join(map(str, train_year))}_{test_year}'
        run_benchmark(data_config, extra=f'rolling_{train_years_str}_{test_years_str}_0701')


if __name__ == "__main__":

    # default:
    data_config = {
        "dataset_name": "cr2",  # !
        "sent_num": "10",
        "kw_num": "20",
        # "sent_num": "20",
        # "kw_num": "40",
        "n_class": "6",  # !
        # "n_class": "2",  # !
        "split_method": "mixed",
        "train_years": "-1",
        "test_years": "-1",
    }
    data_config["data_path"] = f"./data/{data_config['dataset_name']}_cls{data_config['n_class']}_" \
                               f"{data_config['split_method']}_" \
                               f"st{data_config['sent_num']}_kw{data_config['kw_num']}"

    print(data_config)

    # tested and found best pretrain_epochs
    # "cr_cls2": "1",
    # "cr_cls6": "2",
    # "cr2_cls2": "9",
    # "cr2_cls6": "4",
    pretrain_epochs = {
        "cr_cls2": "2",
        "cr_cls6": "2",
        "cr2_cls2": "9",
        "cr2_cls6": "4",
    }

    # loop over all datasets
    for dt in ['cr', 'cr2',]:
        # for dt in ['cr2']:
        # for n_cls in ['2', '6']:
        for n_cls in ['2']:
            data_config["dataset_name"] = dt
            data_config["n_class"] = n_cls
            data_config["data_path"] = f"./data/{data_config['dataset_name']}_cls{data_config['n_class']}_" \
                                       f"{data_config['split_method']}_" \
                                       f"st{data_config['sent_num']}_kw{data_config['kw_num']}"
            # split_dataset(data_config)
            # print(data_config)
            # run_pre_epoch_exps(data_config)
            # run_modality_exps(data_config, pre_epoch=pretrain_epochs[f'{dt}_cls{n_cls}'])
            # run_kw_or_txt_exp(data_config, pre_epoch=pretrain_epochs[f'{dt}_cls{n_cls}'])

            # for pre in [ 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            #     run_rolling_window_exps(data_config, str(pre))

            # run_rolling_window_exps(data_config, pre_epoch=pretrain_epochs[f'{dt}_cls{n_cls}'])
            # run_conv_fusion_exp(data_config, pre_epoch=pretrain_epochs[f'{dt}_cls{n_cls}'])
            # run_benchmark(data_config)
            run_benchmark_rolling_window(data_config)

    # generate_dataset(data_config)

    # split_dataset(data_config)

    # run_pre_epoch_exps(data_config)

    # run_modality_exps(data_config)

    # run_rolling_window_exps(data_config)
