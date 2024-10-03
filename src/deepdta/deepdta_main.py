import logging
import os
from pathlib import Path
from typing import Optional, List

from deepdta.deepdta_model import DeepDTAModel
from deepdta.deepdta_train import DeepDTATrainer
from deepdta.multi_data import MultiData

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def path_to(relative_to_project: str) -> str:
    base_dir = Path(os.path.dirname(__file__)).parent.parent
    return base_dir / relative_to_project

def main(multiple_gpus: Optional[List[int]] = None):

    multi_data = MultiData.load_data_config(path_to("src/configs/sanity_check.json"))
    trainer = DeepDTATrainer(output_dir=path_to("output"), run_tensorboard=True)
    trainer.set_data(multi_data)
    logging.info("Data loaded and set on trainer: %s", multi_data.report())

    deep_dta_model = DeepDTAModel()

    model = deep_dta_model.prepare_model()
    trainer.train(model, epochs=2, batch_size=5)
    metrics = trainer.test()
    print(f"Test report:\n{metrics}")


if __name__  == "__main__":
    main(multiple_gpus=None) #[0, 1, 2, 3])
