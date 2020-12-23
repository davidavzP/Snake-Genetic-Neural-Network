import abc

class Object(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def getPoints(self) -> list:
        raise NotImplementedError


