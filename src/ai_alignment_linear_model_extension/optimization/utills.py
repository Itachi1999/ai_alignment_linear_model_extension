from enum import Enum, auto

class ModelType(Enum):
    EPSILON_LP = auto()
    KT_MIN_LP = auto()
    BTL = auto()
    BTL_HINGE = auto()
    
    @property
    def label(self) -> str:
        return {
            ModelType.EPSILON_LP: "Old LP",
            ModelType.KT_MIN_LP: "New LP",
            ModelType.BTL: "BTL",
            ModelType.BTL_HINGE: "BTL Hinge",
        }[self]