import logging
import random

from ray import tune
from torch.utils.data import Dataset

from data.dataset_registry import dataset_registry
from models.base_model_generator import BaseModelGenerator
from models.model_generator_registry import model_generator_registry
from models.init_fns import INIT_FN_MAP
from utils.experiment_validation import validate_experiments_for_tune
from utils.utils import generate_unique_model_id

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Experiment:
    def __init__(
        self,
        dataset: Dataset,
        hyperparameters: dict,
        model_type: str,
        model: type[BaseModelGenerator],
    ):
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.model = model
        self.model_id = generate_unique_model_id(model, hyperparameters)
        self.model_type = model_type

    def __str__(self):
        return f"Dataset_{self.dataset}_model_{self.model_type}_{self.model_id}"


def compute_trial_arch_seed_order(
    model_types: list,
    model_type_counts: dict,
) -> list[int]:
    """
    Fixed run order: all TRANSFORMER arch_seeds, then RNN, then CNN (ascending arch_seed within each block).

    Seed ranges follow `model_types` / `model_type_counts` (contiguous arch_seed from 1), same as `_model_type_mapping`.
    """
    ordered_types = list(model_types)
    counts = [model_type_counts[mt] for mt in ordered_types]
    cur = 1
    by_type: dict[str, list[int]] = {mt: [] for mt in ordered_types}
    for mt, c in zip(ordered_types, counts):
        by_type[mt] = list(range(cur, cur + c))
        cur += c

    tr = by_type.get("TRANSFORMER", [])
    rnn = by_type.get("RNN", [])
    cnn = by_type.get("CNN", [])

    out = tr + rnn + cnn
    logging.info(
        "📊 Trial order: %d TRANSFORMER, then %d RNN, then %d CNN (ascending arch_seed)",
        len(tr),
        len(rnn),
        len(cnn),
    )
    return out


class ExperimentConfig:
    def __init__(self, config: dict, seed: int, test_mode: bool = False):
        self.dataset = config["dataset"]  # Changed from 'datasets' to 'dataset'
        self.model_types = config["model_types"]
        self.hyperparameters: dict = config["hyperparameters"]
        self.num_models = config["num_models"]
        self.model_type_counts = config["model_type_counts"]
        self.seed = seed
        self.test_mode = test_mode
        random.seed(self.seed)

    def __str__(self):
        # Create a string of the hyperparameters
        hyperparameters_str = ""
        for k, v in self.hyperparameters.items():
            if k == "batch_size":
                hyperparameters_str += f"bsz_{v}_"
            elif k == "num_epochs":
                hyperparameters_str += f"eps_{v}_"
            elif k == "learning_rate":
                hyperparameters_str += f"lr_{v}_"
            elif k == "weight_decay":
                hyperparameters_str += f"wd_{v}_"
            elif k == "gamma":
                hyperparameters_str += f"gamma_{v}_"
            elif k == "step_size":
                hyperparameters_str += f"step_{v}_"
            elif k == "init_name":
                hyperparameters_str += f"init_{v}_"

        hyperparameters_str = hyperparameters_str.rstrip(
            "_"
        )  # Remove trailing underscore
        return f"{self.dataset}_{hyperparameters_str}"

    def generate_experiments(self):
        # NOTE: not used with tune runner
        experiments = []
        for model_type in self.model_types:
            # Directly use the provided hyperparameters as there is only one value for each
            hyperparameters = self.hyperparameters

            dataset_class = dataset_registry.get_dataset(
                self.dataset
            )  # Use the single dataset
            data_input_size = dataset_class.config["input_size"]
            data_classes = dataset_class.config["num_classes"]

            model_generator = model_generator_registry.get_generator(model_type)
            completed_model_configs = []

            for _ in range(self.num_models):
                model = model_generator(data_input_size, data_classes).generate_model()
                while (
                    generate_unique_model_id(model, hyperparameters)
                    in completed_model_configs
                ):
                    logging.info(
                        f"Generated duplicate {model_type} model. Regenerating..."
                    )
                    model = model_generator(
                        data_input_size, data_classes
                    ).generate_model()
                completed_model_configs.append(
                    generate_unique_model_id(model, hyperparameters)
                )
                experiment = Experiment(
                    self.dataset, hyperparameters, model_type, model
                )  # Use the single dataset
                experiments.append(experiment)

        random.shuffle(experiments)
        return experiments

    def build_search_space(
        self,
        *,
        input_size: int | None = None,
        num_classes: int | None = None,
    ) -> dict:
        hp = self.hyperparameters

        validate_experiments_for_tune(
            {
                "dataset": self.dataset,
                "model_types": self.model_types,
                "num_models": self.num_models,
                "hyperparameters": hp,
                "model_type_counts": self.model_type_counts,
            }
        )

        space = {
            "step_size": hp["step_size"],
            "gamma": hp["gamma"],
            "learning_rate": hp["learning_rate"],
            "weight_decay": hp["weight_decay"],
            "batch_size": hp["batch_size"],
            "num_epochs": hp["num_epochs"],
            "init_name": tune.choice(list(INIT_FN_MAP.keys())),
            "init_seed": tune.choice([0]),
        }

        ordered_types = list(self.model_types)
        counts = [self.model_type_counts[mt] for mt in ordered_types]
        total_models = sum(counts)
        assert total_models == self.num_models

        seed_ranges = []
        current_seed = 1
        for count in counts:
            seed_ranges.append((current_seed, current_seed + count - 1))
            current_seed += count

        mapping_payload = {"ordered_types": ordered_types, "seed_ranges": seed_ranges}
        hp["_model_type_mapping"] = mapping_payload
        space["_model_type_mapping"] = mapping_payload

        if input_size is None or num_classes is None:
            raise ValueError(
                "build_search_space requires input_size and num_classes from the dataset."
            )
        arch_list = compute_trial_arch_seed_order(
            self.model_types,
            self.model_type_counts,
        )
        if len(arch_list) != total_models:
            raise RuntimeError(
                f"arch_seed run-order length {len(arch_list)} != total_models {total_models}"
            )
        space["arch_seed"] = tune.grid_search(arch_list)
        logging.warning(
            "📊 arch_seed grid = TRANSFORMER → RNN → CNN (ascending arch_seed within each block); "
            "Ray uses this order as trial priority (FIFO); concurrency is limited only by cluster resources."
        )
        self.model_type_counts = {mt: c for mt, c in zip(ordered_types, counts)}
        logging.info(f"📊 Using model_type_counts: {self.model_type_counts}")
        logging.info(f"📊 Model type order: {ordered_types}")
        logging.info(f"📊 Seed ranges: {seed_ranges}")
        logging.info(f"📊 Total models: {total_models}")
        logging.info(f"📊 Created {total_models} trials via grid_search on arch_seed")
        return space
