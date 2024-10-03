
import json
import pickle
from typing import Dict

import numpy as np


class MultiData:
    """
    A class to hold multiple data sets for training, validation, and testing, loaded from a prepared pickle file.

    The pickle file specified as `datafile` should contain a Dictionary with keys for each data set,
    and values that are numpy arrays of the data.

    The file is only loaded, and split when any of the properties: training, validation, or test are accessed.
    """
    def __init__(self, data_file: str, training_start: int, training_size: int, validation_size: int, test_size: int):
        self._data_file = data_file
        self._training_start = training_start
        self._training_size = training_size
        self._validation_size = validation_size
        self._test_size = test_size
        self._training = None
        self._validation = None
        self._test = None
        self._loaded = False

    @property
    def training(self) -> Dict[str, np.ndarray]:
        if self._training is None:
            self._load_and_split()
        return self._training

    @property
    def validation(self) -> Dict[str, np.ndarray]:
        if self._validation is None:
            self._load_and_split()
        return self._validation

    @property
    def test(self) -> Dict[str, np.ndarray]:
        if self._test is None:
            self._load_and_split()
        return self._test

    @property
    def keys(self):
        return self.training.keys()

    @staticmethod
    def load_data_config(config_file: str):
        with open(config_file, 'r') as file:
            data = json.load(file)  # Load JSON into a dictionary
        return MultiData(**data)  # Use the dictionary to create a dataclass instance

    def save_data_config(self, config_file: str):
        with open(config_file, 'w') as file:
            json.dump({
                'data_file': self._data_file,
                'training_start': self._training_start,
                'training_size': self._training_size,
                'validation_size': self._validation_size,
                'test_size': self._test_size,
            }, file)

    def _split(self, encoded_data: Dict):
        training_end = self._training_start + self._training_size
        validation_end = training_end + self._validation_size
        test_end = validation_end + self._test_size

        # Split the dict of encoded x_prot, x_met, y_train into 3 dicts with train, validation, and test sets
        self._training = {key: np.array(data[self._training_start:training_end]) for key, data in encoded_data.items()}
        self._validation = {key: np.array(data[training_end:validation_end]) for key, data in encoded_data.items()}
        self._test = {key: np.array(data[validation_end:test_end]) for key, data in encoded_data.items()}

    def _load_and_split(self):
        if not self._loaded:
            self._loaded = True
            with open(self._data_file, 'rb') as file:
                data = pickle.load(file)
            self._split(data)

    @staticmethod
    def _format_shapes(data_dict):
        return '  '.join([f"{key}: {str(data.shape):<{15}}" for key, data in data_dict.items()])

    def report(self) -> str:
        return (
            f"Multiple datasets with keys: {self.keys}\n and shapes:\n"
            f"  training:    {self._format_shapes(self.training)}\n"
            f"  validation:  {self._format_shapes(self.validation)}\n"
            f"  test:        {self._format_shapes(self.test)}\n"
        )

