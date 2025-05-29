# Credit Rating Multi-Modal Project

This repository contains the official implementation for the paper:

**Lu, S., Zhang, X., Su, Y., Liu, X., & Yu, L. (2025). Efficient multimodal learning for corporate credit risk prediction with an extended deep belief network. Annals of Operations Research, 1-38.**

Paper Link: https://link.springer.com/article/10.1007/s10479-025-06612-w

If you find this code useful for your research, please consider citing our paper:

```bibtex
@article{lu2025efficient,
  title={Efficient multimodal learning for corporate credit risk prediction with an extended deep belief network},
  author={Lu, S. and Zhang, X. and Su, Y. and Liu, X. and Yu, L.},
  journal={Annals of Operations Research},
  pages={1--38},
  year={2025},
  publisher={Springer}
}
```

## Setup

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The project requires Python 3.8+ and the libraries listed in `requirements.txt`.

## Data Processing

The project processes SEC filings and financial data for credit rating prediction. The data processing pipeline consists of:

1. Text summarization from SEC filings
2. Keyword extraction
3. Dataset splitting

Run the data generation process with:

```bash
python run_exps.py
```

Or use the shell script:

```bash
./run.sh
```

## Running Experiments

### Basic Usage

The main experiment runner is `main_runner.py`, which supports the following tasks:
- `pretrain`: Pre-train the multi-modal model
- `fine_tune`: Fine-tune a pre-trained model
- `fine_tune_from_scratch`: Train the model from scratch without pre-training
- `explain_visual`: Generate visualizations to explain the model's predictions

### Detailed Parameter Explanation

#### Main Parameters for `main_runner.py`:

```
--task                          Task type: pretrain, fine_tune, fine_tune_from_scratch, or explain_visual
--bert_model_name               Name of the pretrained BERT model (e.g., prajjwal1/bert-tiny, bert-base-uncased)
--use_hf_pretrained_bert_in_pretrain  Whether to use HuggingFace pretrained BERT in pretraining (true/false)
--freeze_bert_params            Whether to freeze BERT parameters during training (true/false)
--use_modality                  Modalities to use, comma-separated (e.g., num,cat,text)
--modality_fusion_method        Method for fusing modalities (e.g., concat, conv)
--text_cols                     Text columns to use, comma-separated (e.g., secText,secKeywords)
--per_device_train_batch_size   Batch size for training
--num_train_epochs              Number of training epochs
--patience                      Early stopping patience
--dataset_name                  Name of the dataset (e.g., cr, cr2)
--dataset_info                  Additional dataset information
--data_path                     Path to the dataset
--output_dir                    Directory to save output files
--logging_dir                   Directory to save logs
--save_excel_path               Path to save results as Excel file
--pretrained_model_dir          Directory containing pretrained model (for fine_tune task)
--small_params                  Whether to use smaller model parameters (true/false)
```

#### Parameters for Data Processing Scripts:

```
--sentences_num                 Number of sentences to extract in text summarization
--keywords_num                  Number of keywords to extract
--prev_step_sentences_num       Number of sentences from previous step
--prev_step_keywords_num        Number of keywords from previous step
--n_class                       Number of classes for classification
--split_method                  Method for splitting dataset (e.g., mixed, rolling_window)
--train_years                   Years to use for training
--test_years                    Years to use for testing
```

### Using the Run Script

The simplest way to run experiments is through the provided shell script:

```bash
./run.sh
```

This script will:
1. Generate the dataset with text summarization and keyword extraction
2. Split the dataset using the specified method
3. Run the experiments with different modalities (numerical, categorical, text) and training strategies

### Custom Experiments

For more control, you can use `run_exps.py` to run specific experiments:

```bash
python run_exps.py
```

The script contains various functions for running different types of experiments:
- `run_pre_epoch_exps`: Test different pre-training epochs
- `run_modality_exps`: Test different combinations of modalities
- `run_conv_fusion_exp`: Test different fusion methods
- `run_rolling_window_exps`: Run experiments with rolling window validation
- `run_benchmark`: Run benchmark comparisons

### Model Explanation

To generate visualizations that explain the model's predictions:

```bash
./run_explain.sh
```

This will generate visualizations showing which features are most important for the model's predictions, stored in the project directory.

## Project Structure

- `crmm/`: Core model implementation
  - `data_acquisition/`: Scripts for data processing and acquisition
  - `dataset/`: Dataset implementations
  - `models/`: Model implementations including the extended Deep Belief Network
- `exps/`: Output directory for experiment results
- `excel/`: Excel files with experiment results
- `main_runner.py`: Main script for running experiments
- `run_exps.py`: Python script for running batches of experiments
- `run.sh`: Shell script for running predefined experiments
- `run_explain.sh`: Shell script for generating model explanations

## Configuration Example

Here's an example configuration for running an experiment:

```bash
python main_runner.py --task fine_tune_from_scratch \
                      --bert_model_name prajjwal1/bert-tiny \
                      --use_hf_pretrained_bert_in_pretrain true \
                      --freeze_bert_params false \
                      --use_modality num,cat,text \
                      --modality_fusion_method conv \
                      --text_cols secKeywords \
                      --per_device_train_batch_size 300 \
                      --num_train_epochs 200 \
                      --patience 1000 \
                      --dataset_name cr \
                      --data_path ./data/cr_cls2_mixed_st10_kw20 \
                      --output_dir ./exps/my_experiment/output \
                      --logging_dir ./exps/my_experiment/logging \
                      --save_excel_path ./excel/my_results.xlsx
```

## Results

Experiment results are saved in Excel files in the `excel/` directory, including metrics for validation and test sets.


