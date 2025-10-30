from abc import ABC, abstractmethod
from omegaconf import DictConfig

class Procedure(ABC):
    def __init__(self, config: DictConfig):
        self.config = config