import logging
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.python.distribute.distribute_lib import Strategy

# Define the protein vocabulary: an integer token for each amino acid
CHAR_PROT_SET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21,
    "V": 22, "Y": 23, "X": 24,
    "Z": 25
}

# Define the SMILES vocabulary: an integer token for each SMILES character (typically an atom or a bond)
CHAR_SMILE_SET = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34,
    ".": 2, "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5,
    "7": 38, "6": 6, "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
    "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
    "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
    "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
    "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64, '*': 65
}


class DeepDTAModel:

    def __init__(
            self,
            char_smi_set_size: int = len(CHAR_SMILE_SET),
            char_prot_set_size: int = len(CHAR_PROT_SET),
            prot_len: int = 1198,
            smile_len: int = 176,
            num_filters: int = 32,
            gpu_list: list[int] | None = None,
    ):
        self._char_smi_set_size = char_smi_set_size
        self._char_prot_set_size = char_prot_set_size
        self._prot_len = prot_len
        self._smile_len = smile_len
        self._num_filters = num_filters
        self._gpu_list = gpu_list
        self._interaction_model: Model | None = None


    @staticmethod
    def _build_convolutional_layers(input_layer, base_filters: int, kernel_sizes: list[int]):
        x = input_layer
        for i, kernel_size in enumerate(kernel_sizes):
            num_filters = base_filters * (i + 1) # 1, then 2, then 3, etc.
            x = Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', padding='valid', strides=1)(x)
            logging.debug("Convolutional layer %d: %s", i + 1, x.shape)
        x = GlobalMaxPooling1D()(x)
        logging.debug("Global Max Pooling layer %s", x.shape)
        return x

    def _build(self):
        x_prot = Input(shape=(self._prot_len,), dtype='int32')
        x_met = Input(shape=(self._smile_len,), dtype='int32')  ### Buralar flagdan gelmeliii (turkish from deepDTA: these should be flags)

        # Embed the proteins in the latent space:
        # _char_prot_set_size is the size of the vocabulary, i.e., the number of unique tokens (amino acids) in the protein sequences.
        # 128 is the dimension of the dense embedding vectors.
        # _prot_len is the length of the input sequences - i.e. the protein sequences in the dataset
        encode_protein = Embedding(input_dim=self._char_prot_set_size, output_dim=128, input_length=self._prot_len)(x_prot)
        logging.debug("Protein embedding shape: %s", encode_protein.shape)
        encode_protein = self._build_convolutional_layers(encode_protein, self._num_filters, [4, 8, 12])

        encode_smiles = Embedding(input_dim=self._char_smi_set_size, output_dim=128, input_length=self._smile_len)(x_met)
        encode_smiles = self._build_convolutional_layers(encode_smiles, self._num_filters, [4, 6, 8])
        logging.info(
            "Concatenating protein and smiles layers: Protein shape: %s  Smiles shape: %s",
            encode_protein.shape, encode_smiles.shape
        )

        encode_interaction = tf.keras.layers.concatenate([encode_smiles, encode_protein], axis=-1)  # merge.Add()([encode_smiles, encode_protein])

        # Fully connected
        FC1 = Dense(1024, activation='relu')(encode_interaction)
        FC2 = Dropout(0.1)(FC1)
        FC2 = Dense(1024, activation='relu')(FC2)
        FC2 = Dropout(0.1)(FC2)
        FC2 = Dense(512, activation='relu')(FC2)

        # And add a logistic regression on top
        predictions = Dense(1, activation='sigmoid')(FC2)  # kernel_initializer='normal"
        self._interaction_model = Model(inputs=[x_prot, x_met], outputs=[predictions])
        return self._interaction_model

    def _init_environment(self) -> Strategy | None:
        seed_value = 42
        np.random.seed(seed_value)
        random.seed(seed_value)
        tf.random.set_seed(seed_value)

        os.environ['PYTHONHASHSEED'] = str(seed_value)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

        logging.info("Num GPUs Available: %d", len(physical_devices))

        strategy: Strategy | None  = None

        if self._gpu_list:
            devices = []
            for gpu in self._gpu_list:
                devices.append('/gpu:' + str(gpu))
            strategy = tf.distribute.MirroredStrategy(devices=devices)
            os.environ["CUDA_VISIBLE_DEVICES"] = ''
        else:
            # Get the GPU device name.
            device_name = tf.test.gpu_device_name()
            # The device name should look like the following:
            if device_name == '/device:GPU:0':
                logging.info("Using GPU: %s", device_name)
            else:
                raise SystemError(f'Expected GPU device not found. Only found {device_name}')

            os.environ["CUDA_VISIBLE_DEVICES"] = device_name
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        return strategy

    def prepare_model(
            self,
            print_summary: bool = True,
            optimizer: str = "adam",
            loss: str = "binary_crossentropy",
            metrics: list = ["accuracy"],
    ):
        logging.info("Building and compiling a DeepDTA model using the following hyperparameters: "
                     f"optimizer: {optimizer}, loss: {loss}, metrics: {metrics}")
        strategy = self._init_environment()
        if strategy:
            with strategy.scope():
                interaction_model = self._build()
                interaction_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        else:
            interaction_model = self._build()
            interaction_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self._interaction_model = interaction_model
        if print_summary:
            print(interaction_model.summary())
        return interaction_model
