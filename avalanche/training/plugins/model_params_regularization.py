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


class RegularizationPlugin(SupervisedPlugin):
    def __init__(self,alpha):
        super().__init__()

        self.params_current = None
        self.params_last_t = None
        self.alpha = alpha
        self.file0 = open(get_project_path()+"train_logs/momentum_classifier0.txt",mode="w+")
        self.file1 = open(get_project_path()+"train_logs/momentum_extractor.txt",mode="w+")


    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        if self.params_last_t is None: # 第一个exp
            self.params_last_t = copy.deepcopy(strategy.model.extractor.state_dict())
        else: # 后面的exp，需要momentum update
            self.params_last_t = copy.deepcopy(strategy.model.extractor.state_dict()) # 先将当前模型的权重保存
            self.params_current = self.momentum_update(self.params_momentum,self.params_task)
            strategy.model.extractor.load_state_dict(self.params_momentum)
            print("reaching 1.2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # 保存extractor某一层的参数
        # extrac = strategy.model.extractor.state_dict()
        # print("-"*80,file=self.file1,flush=True)
        # print(extrac['6.0.conv1.weight'],file=self.file1,flush=True)



            # self.momentum_model.extractor.load_state_dict(self.params_momentum)

    # def before_eval(
    #     self, strategy: "SupervisedTemplate", *args, **kwargs
    # ):
    #     print("EVAL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     if self.params_momentum is not None:
    #         strategy.model.extractor.load_state_dict(self.params_momentum)

    # def after_eval(self, strategy: "SupervisedTemplate", *args, **kwargs):
    #     if self.momentum_model is not None:
    #         strategy.model.extractor.load_state_dict(self.params_task)

    def momentum_update(self,params_momentum,params_current):
        assert params_momentum.keys() == params_current.keys()
        print("Updating..............................................................")
        for name in params_momentum.keys():
            params_momentum[name] = torch.add(torch.mul(params_momentum[name],self.alpha),torch.mul(params_current[name],(1-self.alpha)))
        return params_momentum
