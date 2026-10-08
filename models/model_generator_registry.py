from models.base_model_generator import BaseModelGenerator

from .cnn_model_generator import CNNModelGenerator
from .rnn_model_generator import RNNModelGenerator
from .tinyhar_model_generator import TinyHARModelGenerator
from .tinyhar_model_exact import TinyHARReferenceModelGenerator
from .transformer_model_generator import TransformerModelGenerator


class ModelGeneratorRegistry:
    def __init__(self):
        self.generators = {}

    def register(self, name, generator_class: type[BaseModelGenerator]):
        self.generators[name] = generator_class

    def get_generator(self, name: str) -> type[BaseModelGenerator] | None:
        return self.generators.get(name)


model_generator_registry = ModelGeneratorRegistry()
model_generator_registry.register("CNN", CNNModelGenerator)
model_generator_registry.register("RNN", RNNModelGenerator)
model_generator_registry.register("TinyHAR_RANDOM", TinyHARModelGenerator)
model_generator_registry.register("TinyHAR_REFERENCE", TinyHARReferenceModelGenerator)
model_generator_registry.register("TRANSFORMER", TransformerModelGenerator)