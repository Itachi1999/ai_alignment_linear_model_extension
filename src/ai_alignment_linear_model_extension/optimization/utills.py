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
            ModelType.EPSILON_LP: "LP2 (Theta Only)",
            ModelType.KT_MIN_LP: "LP3 (Theta Only)",
            ModelType.BTL: "BTL",
            ModelType.BTL_HINGE: "BTL Hinge",
            ModelType.BTL_LINEAR: "Linear BTL",
            ModelType.BTL_LINEAR_HINGE: "BTL Linear Hinge",
        }[self]