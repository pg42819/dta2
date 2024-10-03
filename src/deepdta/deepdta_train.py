import logging
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
from keras.src.callbacks import TensorBoard
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from deepdta.multi_data import MultiData


class DeepDTATrainer:

    def __init__(
            self,
            output_dir: str | Path,
            run_tensorboard: bool = False,
    ):
        self._output_dir = Path(output_dir).resolve()
        self._training_checkpoints_dir = None
        self._run_tensorboard = run_tensorboard
        self._tensorboard_log_dir = None
        self._final_weights_file: str | None = None
        self._training_history_file: str | None = None
        self._data = None
        self._trained = False
        self._model = None
        self._tested = False


    @property
    def final_weights_file(self) -> Optional[str]:
        if not self._trained:
            raise RuntimeError("The model must be trained with a call to train() before accessing the trained weights file.")
        return self._final_weights_file

    @property
    def training_history_file(self) -> Optional[str]:
        if not self._trained:
            raise RuntimeError("The model must be trained with a call to train() before accessing the training history file.")
        return self._training_history_file

    @property
    def y_pred_test(self) -> Optional[np.ndarray]:
        if not self._tested:
            raise RuntimeError("The model must be tested with a call to test() before accessing the test predictions.")
        return self._y_pred_test

    @property
    def multi_data(self):
        if self._data is None:
            raise RuntimeError("The multi_data property must be set before training or testing.")
        return self._data

    def set_data(self, multi_data: Union[MultiData, str]):
        if isinstance(multi_data, str):
            multi_data = MultiData.load_data_config(multi_data)
        elif isinstance(multi_data, MultiData):
            multi_data = multi_data
        else:
            raise ValueError("multi_data must be either a MultiData instance or a path to a JSON config file.")
        self._data = multi_data

    @staticmethod
    def _timestamp_dir(base_dir: Path, base_name: str, ext: str = "") -> Path:
        if not base_dir.exists():
            raise FileNotFoundError(f"Output directory {base_dir} does not exist.")
        time_stamp = "{:%Y_%m_%d_%H_%M}".format(datetime.now())  # hour only
        filename = f"{base_name}_{time_stamp}{ext}"
        return base_dir / filename

    def _init_training_run(self):
        self._training_checkpoints_dir = self._timestamp_dir(self._output_dir, "training_checkpoints")
        self._training_checkpoints_dir.mkdir()
        logging.info("Training run output dir created at %s", self._training_checkpoints_dir)
        if self._run_tensorboard:
            self._tensorboard_log_root = self._output_dir / "tensorboard_logs"
            self._tensorboard_log_root.mkdir(exist_ok=True)
            self._tensorboard_log_dir = self._timestamp_dir(self._tensorboard_log_root, "training_run")
            logging.info("TensorBoard logs will be saved to %s", self._tensorboard_log_dir)
            print(f"Use the following command to start TensorBoard:\n tensorboard --logdir {self._tensorboard_log_root} serve")
            # We can start it as a subprocess but need to add a way to stop it
            # tensorboard_command = ["tensorboard", "--logdir", str(self._tensorboard_log_dir), "serve"]
            # Could use subprocess.PIPE here but by sending to sys.stdout, we see the URL for the server in the console
            # tensorboard_process = subprocess.Popen(tensorboard_command, stdout=sys.stdout, stderr=sys.stderr)
            # print(f"Running tensorboard with:\ntensorboard --logdir {self._tensorboard_log_dir} serve")

    def _output_path(self, base_name, ext="") -> Path:
        return self._timestamp_dir(self._training_checkpoints_dir, base_name, ext)

    def train(
            self,
            model: Model,
            epochs: int = 5,
            batch_size: int = 256,
    ) -> Tuple[str, str]:
        self._init_training_run()
        logging.info("Starting training run with %d epochs and batch size %d", epochs, batch_size)
        keys = self.multi_data.keys
        if not 'x_prot' in keys or not 'x_met' in keys or not 'y_train' in keys:
            raise ValueError(f"MultiData must have datasets keyed as 'x_prot', 'x_met', and 'y_train'")

        checkpoint_path = self._output_path("weights_checkpoints", "e{epoch:02d}_vloss-{val_loss:.2f}.keras")
        ext = "e{epoch:02d}_vloss-{val_loss:.2f}.keras"
        train_history_filename = self._output_path(f"train_history", ".pkl")

        keras_callbacks = [
            EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5),
            ModelCheckpoint(checkpoint_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True),
        ]

        if self._run_tensorboard:
            keras_callbacks.append(TensorBoard(log_dir=self._tensorboard_log_dir, histogram_freq=1))

        training, validation = self.multi_data.training, self.multi_data.validation
        inputs = [training['x_prot'], training['x_met']]
        labels = training['y_train']
        validation_inputs = [validation['x_prot'], validation['x_met']]
        validation_labels = validation['y_train']

        history = model.fit(
            x=inputs,
            y=labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(validation_inputs, validation_labels),
            callbacks=keras_callbacks
        )

        with open(train_history_filename, "wb") as file_pi:
            pickle.dump(history.history, file_pi)
            logging.info(f"Training history saved to {train_history_filename}")

        self._final_weights_file = self._output_path("final_model", ext=".keras")
        model.save(self._final_weights_file)
        logging.info(f"Model weights saved to final_model in {self._final_weights_file}")
        self._training_history_file = train_history_filename
        self._trained = True
        self._model = model
        return model

    def test(
            self,
            model: Model | None = None,
            weights_file: str | Path | None = None,
    ):
        # TODO move the weights_file into a directory
        # TODO add a predict method to the model
        if model is None:
            if self._model is None:
                raise RuntimeError("The model must be trained with a call to train() before testing.")
            else:
                model = self._model

        keys = self.multi_data.test.keys()
        if not 'x_prot' in keys or not 'x_met' in keys or not 'y_train' in keys:
            raise ValueError(f"MultiData must have datasets keyed as 'x_prot', 'x_met', and 'y_train'")

        x_prot_test = self.multi_data.test['x_prot']
        x_met_test = self.multi_data.test['x_met']

        if weights_file is not None:
            logging.info("Loading model weights from %s", weights_file)
            model.load_weights(weights_file)

        # using model.predict instead of model.evaluate because we use it for custom metric calculations below
        y_pred_test = model.predict([x_prot_test, x_met_test], verbose=0)
        y_test = self.multi_data.test['y_train']


        logging.info("Calculating metric for y_test: %s and y_pred_test: %s", y_test, y_pred_test)

        yhat_classes = np.where(y_pred_test > 0.5, 1, y_pred_test)
        yhat_classes = np.where(yhat_classes < 0.5, 0, yhat_classes).astype(np.int64)

        logging.info("Filtered to yhat_classes: %s", yhat_classes)

        metrics = {}  # keep metrics in a dict

        # accuracy: (tp + tn) / (p + n)
        metrics["accuracy"] = accuracy_score(y_test, yhat_classes)

        # precision tp / (tp + fp)
        metrics["precision"] = precision_score(y_test, yhat_classes)

        # recall: tp / (tp + fn)
        metrics["recall"] = recall_score(y_test, yhat_classes)

        # f1: 2 tp / (2 tp + fp + fn)
        metrics["f1"] = f1_score(y_test, yhat_classes)
        metrics["cohens_kappa"] = cohen_kappa_score(y_test, yhat_classes)
        # matthews correlation coefficient
        metrics["mcc"] = matthews_corrcoef(y_test, yhat_classes)
        metrics["confusion_matrix"] = confusion_matrix(y_test, yhat_classes)
        self._tested = True

        return metrics

