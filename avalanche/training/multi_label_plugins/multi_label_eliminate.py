from typing import Optional, TYPE_CHECKING

import torch

from avalanche.benchmarks.utils import concat_classification_datasets
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer,
)

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate
from avalanche.benchmarks.utils.collate_functions import mixup_classification_collate_mbatches_fn
import copy
from tools.get_path import get_project_path


class EliminatePlugin(SupervisedPlugin):
    def __init__(self,alpha):
        super().__init__()

        self.params_momentum = None
        self.params_task = None
        self.alpha = alpha


    def after_forward(
        self, strategy: "SupervisedTemplate", *args, **kwargs
    ):
        pred_y,true_y = strategy.mb_output,strategy.mb_y
