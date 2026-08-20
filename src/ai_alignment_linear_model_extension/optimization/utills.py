from enum import Enum, auto

class ModelType(Enum):
    EPSILON_LP = auto()
    KT_MIN_LP = auto()
    BTL = auto()
    BTL_HINGE = auto()
    BTL_LINEAR = auto()
    BTL_LINEAR_HINGE = auto()
    
    @property
    def label(self) -> str:
        return {
            ModelType.EPSILON_LP: "Old LP",
            ModelType.KT_MIN_LP: "New LP",
            ModelType.BTL: "BTL",
            ModelType.BTL_HINGE: "BTL Hinge",
            ModelType.BTL_LINEAR: "BTL Linear",
            ModelType.BTL_LINEAR_HINGE: "BTL Linear Hinge",
        }[self]