from enum import Enum, auto

class ModelType(Enum):
    EPSILON_LP = auto()
    KT_MIN_LP = auto()
    WEIGHTED_KT_MIN_LP = auto()
    BORDA_KT_MIN_LP = auto()
    BTL = auto()
    BTL_HINGE = auto()
    BTL_LINEAR = auto()
    BTL_LINEAR_HINGE = auto()
    
    @property
    def label(self) -> str:
        return {
            ModelType.EPSILON_LP: r"LP2 ($\theta$ Only)",
            ModelType.KT_MIN_LP: r"LP3 ($\theta$ Only)",
            ModelType.WEIGHTED_KT_MIN_LP: r"Weighted LP3 ($\theta$ Only)",
            ModelType.BORDA_KT_MIN_LP: r"Borda LP3 ($\theta$ Only)",
            ModelType.BTL: "BTL",
            ModelType.BTL_HINGE: "BTL Hinge",
            ModelType.BTL_LINEAR: "Linear BTL",
            ModelType.BTL_LINEAR_HINGE: "BTL Linear Hinge",
        }[self]