from abc import ABC, abstractmethod
from omegaconf import DictConfig


class Procedure(ABC):
    registry = {}
    def __init__(self, config: DictConfig):
        self.config = config

    @classmethod
    def load(cls, name, config: DictConfig):
        if name not in cls.registry:
            raise ValueError(f"Unknown procedure: '{name}'.")
        return cls.registry[name](config)

def register(name):
    def wrapper(subclass):
        Procedure.registry[name] = subclass
        return subclass
    return wrapper