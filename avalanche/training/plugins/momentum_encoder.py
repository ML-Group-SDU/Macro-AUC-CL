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
from tools.get_path import get_project_path_and_name


class MomentumPlugin(SupervisedPlugin):
    """
    Experience replay plugin.

    Handles an external memory filled with randomly selected
    patterns and implementing `before_training_exp` and `after_training_exp`
    callbacks.
    The `before_training_exp` callback is implemented in order to use the
    dataloader that creates mini-batches with examples from both training
    data and external memory. The examples in the mini-batch is balanced
    such that there are the same number of examples for each experience.

    The `after_training_exp` callback is implemented in order to add new
    patterns to the external memory.

    The :mem_size: attribute controls the total number of patterns to be stored
    in the external memory.

    :param batch_size: the size of the data batch. If set to `None`, it
        will be set equal to the strategy's batch size.
    :param batch_size_mem: the size of the memory batch. If
        `task_balanced_dataloader` is set to True, it must be greater than or
        equal to the number of tasks. If its value is set to `None`
        (the default value), it will be automatically set equal to the
        data batch size.
    :param task_balanced_dataloader: if True, buffer data loaders will be
            task-balanced, otherwise it will create a single dataloader for the
            buffer samples.
    :param storage_policy: The policy that controls how to add new exemplars
                           in memory
    """

    def __init__(self,alpha):
        super().__init__()

        self.params_momentum = None
        self.params_task = None
        self.alpha = alpha
        self.file0 = open(get_project_path_and_name()[0] + "train_logs/momentum_classifier0.txt",mode="w+")
        self.file1 = open(get_project_path_and_name()[0] + "train_logs/momentum_extractor.txt",mode="w+")


    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):

        #  保存classifier_0的参数
        # params_classifier0 = strategy.model.classifier.classifiers["0"].state_dict()
        # print("-"*80, file=self.file0,flush=True)
        # for name,param in params_classifier0.items():
        #     print(name,param,file=self.file0,flush=True)

        if self.params_momentum is None: # 第一个exp
            self.params_momentum = copy.deepcopy(strategy.model.extractor.state_dict())
        else: # 后面的exp，需要momentum update
            self.params_task = copy.deepcopy(strategy.model.extractor.state_dict()) # 先将当前模型的权重保存
            self.params_momentum = self.momentum_update(self.params_momentum,self.params_task)
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
