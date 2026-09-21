from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from enum import Enum, auto


def setup_plot() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=1.1,
    )


def style_axes(axis: Axes) -> None:
    axis.grid(axis="y", alpha=0.25)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


class ParameterNameMapping(Enum):
    LAMBDA_LP3 = auto()
    NORM_PHI = auto()
    FEATURE_DIMENSION = auto()
    QUESTION_NUMBER = auto()
    TRIAL = auto()
    ALTERNATIVES = auto()
    
    @property
    def label(self) -> str:
        return {
            ParameterNameMapping.LAMBDA_LP3: r"Lambda ($\lambda$)",
            ParameterNameMapping.FEATURE_DIMENSION: r"Feature Dimension ($d$)",
            ParameterNameMapping.ALTERNATIVES: r"Number of Alternatives ($m$)",
            ParameterNameMapping.NORM_PHI: r"Dispersion ($\phi$)",
            ParameterNameMapping.QUESTION_NUMBER: r"Question Number",
            ParameterNameMapping.TRIAL: "Trials"
        }[self]
    