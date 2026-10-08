from abc import ABC, abstractmethod

class BaseModelGenerator(ABC):
    @abstractmethod
    def __init__(self, input_size: int, num_classes: int):
        self.input_size = input_size
        self.num_classes = num_classes

    @abstractmethod
    def generate_model(self):
        pass