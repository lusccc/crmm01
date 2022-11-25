from collections import OrderedDict

from transformers.models.auto.auto_factory import _BaseAutoModelClass

from models.category_autoencoder import CategoryAutoencoderConfig, CategoryAutoencoder
from models.multi_modal_dbn_bk2 import MultiModalDBNConfig, MultiModalDBN
from models.numerical_autoencoder import NumericalAutoencoder, NumericalAutoencoderConfig


class AutoModelForCrmm(_BaseAutoModelClass):
    _model_mapping = OrderedDict(
        [
            (NumericalAutoencoderConfig, NumericalAutoencoder),
            (CategoryAutoencoderConfig, CategoryAutoencoder),
            (MultiModalDBNConfig, MultiModalDBN),
        ]
    )


