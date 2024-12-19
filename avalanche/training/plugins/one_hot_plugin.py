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


class OneHotPlugin(SupervisedPlugin):
    def __init__(self):
        super().__init__()
        self.params_current = None
        self.params_last_t = None

    def before_training_iteration(
        self, strategy, *args, **kwargs
    ):
        # device = strategy
        targets = strategy.mbatch[-2]
        one_hots = torch.zeros(size=[targets.shape[0],strategy.model.classifier.out_features])

        for index, target in enumerate(targets):
            one_hots[index, target] = 1
        strategy.mbatch[-2] = one_hots.to(strategy.device)

    def before_eval_iteration(
        self, strategy, *args, **kwargs
    ):
        # device = strategy
        targets = strategy.mbatch[-2]
        one_hots = torch.zeros(size=[targets.shape[0], strategy.model.classifier.out_features])
        for index, target in enumerate(targets):
            one_hots[index, target] = 1
        strategy.mbatch[-2] = one_hots.to(strategy.device)