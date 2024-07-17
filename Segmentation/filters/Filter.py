from abc import ABC, abstractmethod

class Filter(ABC):
    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def filter(self, image):
        pass
