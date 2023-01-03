import os.path
from typing import Tuple, Optional

import torch
from learnergy.core import Dataset
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import PreTrainedModel, PretrainedConfig, EvalPrediction

import analysis
from metrics import calc_classification_metrics
from models.params_exposed_dbn import ParamsExposedDBN
from multimodal_transformers.model.layer_utils import calc_mlp_dims, MLP, hf_loss_func
import numpy as np
from transformers.utils import logging
from utils import utils

logger = logging.get_logger('transformers')


class MultiModalDBNConfig(PretrainedConfig):
    model_type = 'multi_modal_dbn'

    modality_list = ('num', 'cat')

    def __init__(self,
                 dbn_train_epoch=10,
                 dbn_dataset_dir='./data/cr_sec_ae_embedding',
                 dbn_model_save_dir=None,
                 pretrained_dbn_model_dir=None,
                 tabular_config=None,
                 use_modality=modality_list,
                 use_gpu=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.dbn_train_epoch = dbn_train_epoch
        self.dbn_dataset_dir = dbn_dataset_dir
        self.dbn_model_save_dir = dbn_model_save_dir
        self.pretrained_dbn_model_dir = pretrained_dbn_model_dir
        self.tabular_config = tabular_config
        self.use_modality = {
            'num': True if 'num' in use_modality else False,
            'cat': True if 'cat' in use_modality else False,
        }
        self.use_gpu = use_gpu


class EmbeddingDataset(TorchDataset):

    def __init__(self, data, targets):
        super(EmbeddingDataset, self).__init__()
        self.data = data
        self.targets = targets

    def __getitem__(self, idx: int):
        item_data = {}
        # {modal_name:emb_dt['train'] for modal_name, emb_dt in emb_datasets.items()}, label_train
        for modal_name, emb in self.data.items():
            item_data[modal_name] = emb[idx]
        item_data['label'] = self.targets[idx]
        return item_data

    def __len__(self):
        return len(self.targets)


class MultiModalDBN(PreTrainedModel):
    def __init__(self, config: MultiModalDBNConfig):
        super().__init__(config)
        self.tabular_config = config.tabular_config
        self.dbns = {}

        for modal, use in self.config.use_modality.items():
            if use:
                self.dbns[modal] = ParamsExposedDBN(
                    model="sigmoid",
                    n_visible=64,
                    n_hidden=(128, 256, 128),
                    steps=(1, 1, 1),
                    learning_rate=(0.1, 0.1, 0.1),
                    momentum=(0, 0, 0),
                    decay=(0, 0, 0),
                    temperature=(1, 1, 1),
                    use_gpu=self.config.use_gpu,
                )

        if len(self.config.use_modality) > 1:
            self.dbns['joint'] = ParamsExposedDBN(
                model="sigmoid",
                # n_visible=128,
                n_visible=128+128,
                n_hidden=(128, 128),
                steps=(1, 1),
                learning_rate=(0.1, 0.1),
                momentum=(0, 0),
                decay=(0, 0),
                temperature=(1, 1),
                use_gpu=config.use_gpu,
            )

    def forward(self, data, labels):
        """
        only used for fine-tuning
        """
        for name, dbn in self.dbns.items():
            pass

    def fit_dbn(self, dbn, feature, epoch):
        if isinstance(feature, torch.Tensor):
            feature = feature.clone().cpu().detach()
        feat_dataset = Dataset(feature,
                               # !!!! targets(labels) actually have no effect since it is unsupervised,
                               # here is dummy data
                               targets=torch.from_numpy(np.random.randint(2, size=len(feature))),
                               transform=lambda x: torch.from_numpy(x))  # note targets=feature is just placeholder
        dbn.fit(feat_dataset, batch_size=256, epochs=epoch)
        return dbn

    def fit_dbns(self):
        multi_modal_dbn_outputs = []
        for name, dbn in self.dbns.items():
            # for each modality
            if name != 'joint':
                ae_emb = self.get_embedding_dataset(name)

                # fit dbn of this modality
                self.fit_dbn(dbn, ae_emb, [self.config.dbn_train_epoch for _ in range(len(dbn.models))])
                dbn.save_model(os.path.join(self.config.dbn_model_save_dir, f'{name}_dbn.pt'))

                # get dbn output of this modality
                dbn_output = dbn(ae_emb)
                multi_modal_dbn_outputs.append(dbn_output)
            elif name == 'joint':
                # use addition to fuse
                # multi_modal_dbn_outputs_fused = torch.add(*multi_modal_dbn_outputs)
                multi_modal_dbn_outputs_fused = torch.concatenate(multi_modal_dbn_outputs, dim=1)
                multi_modal_dbn_outputs_fused_sp = F.softplus(multi_modal_dbn_outputs_fused)
                self.fit_dbn(dbn, multi_modal_dbn_outputs_fused_sp,
                             [self.config.dbn_train_epoch for _ in range(len(dbn.models))])
                dbn.save_model(os.path.join(self.config.dbn_model_save_dir, f'{name}_dbn.pt'))

    def manually_fine_tune_only_dbns(self, epoch=10):
        """
        `manually` means we do not use hugging face trainer
        """

        self.analyzer = analysis.Analyzer(print_conf_mat=True)

        # load data
        emb_datasets = {}
        for name, dbn in self.dbns.items():
            if name != 'joint':
                entire_emb_dataset = self.get_embedding_dataset(name)  # note is already tensor data
                # TODO make split ratio in config
                emb_train, emb_val, emb_test = np.split(entire_emb_dataset,
                                                        [int(.8 * len(entire_emb_dataset)),
                                                         int(.9 * len(entire_emb_dataset))])
                emb_datasets[name] = {
                    'train': emb_train,
                    'val': emb_val,
                    'test': emb_test
                }

        labels = np.load(os.path.join(self.config.dbn_dataset_dir, f'labels.npy'))
        unique_label = np.unique(labels)
        n_class = len(unique_label)
        label_train, label_val, label_test = np.split(labels, [int(.8 * len(labels)), int(.9 * len(labels))])
        label_train, label_val, label_test = torch.from_numpy(label_train), \
                                             torch.from_numpy(label_val), \
                                             torch.from_numpy(label_test),

        train_dataset = EmbeddingDataset({modal_name: emb_dt['train'] for modal_name, emb_dt in emb_datasets.items()},
                                         label_train)
        val_dataset = EmbeddingDataset({modal_name: emb_dt['val'] for modal_name, emb_dt in emb_datasets.items()},
                                       label_val)
        test_dataset = EmbeddingDataset({modal_name: emb_dt['test'] for modal_name, emb_dt in emb_datasets.items()},
                                        label_test)

        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=16)
        val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=16)

        # load model
        self.load_fitted_dbns()
        # Creating the Fully Connected layer to append on top of DBNs
        fc = torch.nn.Linear(128, n_class)
        fc = fc.cuda() if self.config.use_gpu else fc
        # Cross-Entropy loss is used for the discriminative fine-tuning
        criterion = nn.CrossEntropyLoss()

        # Creating the optimzers
        optimizers = []
        for _, dbn in self.dbns.items():
            optimizers += [optim.Adam(rbm.parameters(), lr=0.001) for rbm in dbn.models]
        optimizers.append(optim.Adam(fc.parameters(), lr=0.001))

        def perform_forward(batch):
            inputs = {}
            label = None
            for name, data in batch.items():
                if name != 'label':
                    inputs[name] = data.cuda() if self.config.use_gpu else data
                elif name == 'label':
                    label = data.cuda() if self.config.use_gpu else data

            multi_modal_dbn_outputs = []
            joint_dbn_output = None
            for name, dbn in self.dbns.items():
                if name != 'joint':
                    dbn_output = dbn(inputs[name])
                    multi_modal_dbn_outputs.append(dbn_output)
                elif name == 'joint':
                    # multi_modal_dbn_outputs_fused = torch.add(*multi_modal_dbn_outputs)
                    multi_modal_dbn_outputs_fused = torch.concatenate(multi_modal_dbn_outputs, dim=1)
                    multi_modal_dbn_outputs_fused_sp = F.softplus(multi_modal_dbn_outputs_fused)
                    joint_dbn_output = dbn(multi_modal_dbn_outputs_fused_sp)

            # Calculating the fully-connected outputs
            y_pred = fc(joint_dbn_output)
            return y_pred, label

        # For amount of fine-tuning epochs
        for e in range(epoch):
            logger.info(f"Epoch {e + 1}/{epoch}")

            # Resetting metrics
            train_loss, val_acc = 0, 0

            # For every possible batch
            for batch in tqdm(train_loader):
                # For every possible optimizer
                for opt in optimizers:
                    # Resets the optimizer
                    opt.zero_grad()

                y_pred, label = perform_forward(batch)

                # Calculating loss
                loss = criterion(y_pred, label)

                # Propagating the loss to calculate the gradients
                loss.backward()

                # For every possible optimizer
                for opt in optimizers:
                    # Performs the gradient update
                    opt.step()

                # Adding current batch loss
                train_loss += loss.item()

            # Calculate the val accuracy for the model:
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    y_pred, label = perform_forward(batch)

                    # Calculating predictions
                    ep = EvalPrediction(predictions=[y_pred.cpu().numpy(),], label_ids=label.cpu().numpy(),)
                    metric = calc_classification_metrics(ep)
                    logger.info(f'metric: {metric}')

            logger.info(f"Loss: {train_loss / len(train_loader)} | Val Accuracy: {val_acc}")

    def get_embedding_dataset(self, modal_name, activation=True):
        # load data. note fit dbn (pretrain) use the entire dataset
        ae_emb = torch.from_numpy(
            np.load(os.path.join(self.config.dbn_dataset_dir, f'predict_{modal_name}_ae_embedding_results.npy')))
        ae_emb = ae_emb.cuda() if self.config.use_gpu else ae_emb
        ae_emb = F.softplus(ae_emb) if activation else ae_emb
        return ae_emb

    def load_fitted_dbns(self):
        for name, dbn in self.dbns.items():
            dbn.load_model(os.path.join(self.config.pretrained_dbn_model_dir, f'{name}_dbn.pt'))
