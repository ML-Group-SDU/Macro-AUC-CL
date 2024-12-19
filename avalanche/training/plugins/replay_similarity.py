from typing import Optional, TYPE_CHECKING

from torch.utils.data import DataLoader
import torch
from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer,
)
from avalanche.models import FeatureExtractorBackbone, Flatten
if TYPE_CHECKING:
    from avalanche.training.templates.supervised import SupervisedTemplate


class SimilarityReplayPlugin(SupervisedPlugin):
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

    def __init__(
        self,
        mem_size: int = 200,
        batch_size: int = None,
        batch_size_mem: int = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
    ):
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size, adaptive_size=True
            )

    @property
    def ext_mem(self):
        return self.storage_policy.buffer_groups  # a Dict<task_id, Dataset>

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        **kwargs
    ):


        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        similarity_order = get_similarity_order(strategy, self.storage_policy.buffer)

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = strategy.train_mb_size

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = strategy.train_mb_size

        classes_in_this_experience = strategy.experience.classes_in_this_experience
        class_num = len(classes_in_this_experience)
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,
            shuffle=shuffle,
            mixup=strategy.mixup,
            mu=strategy.mu,
            similarity_order=similarity_order,
            mix_times=strategy.mix_times,
            class_num=class_num,
            scenario_mode=strategy.scenario_mode,
        )

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        self.storage_policy.update(strategy, **kwargs)

def get__similarity_order(strategy, memory): # 直接求的task原型
    data = AvalancheDataset(strategy.adapted_dataset)
    memory = AvalancheDataset(memory)
    if len(memory.task_set) > 1:
        mem_datasets = []
        mem_loaders = {}

        for task_id in memory.task_set:
            dataset_ = memory.task_set[task_id]
            mem_datasets.append(dataset_)
            mem_loaders[task_id] = DataLoader(dataset_, batch_size=200, shuffle=True)

        rep_dic = {}
        fea_extra = FeatureExtractorBackbone(strategy.model, "model.extractor")
        for k, loa in mem_loaders.items():
            representation_k = []
            for mb in loa:
                x = mb[0].cuda()
                t = mb[-1].cuda()
                r = fea_extra.forward(x, t)
                r = torch.flatten(r, 1)
                representation_k.append(r)
            representation_k = torch.cat(representation_k, 0)
            representation_k = torch.mean(representation_k, 0)
            rep_dic[k] = representation_k

        data_loader = DataLoader(data, batch_size=200, shuffle=True)
        representation_k = []
        for mb in data_loader:
            x = mb[0].cuda()
            t = mb[-1].cuda()
            r = fea_extra.forward(x, t)
            r = torch.flatten(r, 1)
            representation_k.append(r)
        representation_k = torch.cat(representation_k, 0)
        rep_cur = torch.mean(representation_k, 0)

        cos = torch.nn.CosineSimilarity(0)
        dis_sim = torch.nn.PairwiseDistance(p=2)

        similarity_dic = {}
        for k, v in rep_dic.items():
            similarity_dic[k] = dis_sim(torch.unsqueeze(rep_cur, 0), torch.unsqueeze(v, 0))
        weights_dic = {}
        if len(similarity_dic) >= 2:
            d_order = sorted(similarity_dic.items(), key=lambda x: x[1], reverse=False)  # 升序排列，越小说明越不相似，需要给更大的 weight 值
            task_sim_order = [i[0] for i in d_order]
            return task_sim_order
    else:
        return [0]


def get_similarity_order(strategy, memory):
    data = AvalancheDataset(strategy.adapted_dataset)
    memory = AvalancheDataset(memory)

    if len(memory.task_set) > 1:
        fea_extra = FeatureExtractorBackbone(strategy.model, "model.extractor")
        task_r_dic = {}
        for task_id in memory.task_set:
            dataset_ = memory.task_set[task_id]
            classes = list(set(dataset_.targets))
            c_r_list = []
            for c in classes:
                c_index = [i for i, x in enumerate(dataset_.targets) if x is c]
                c_data = dataset_[c_index]
                x = c_data[0].cuda()
                t = c_data[-1].cuda()
                r = fea_extra.forward(x, t)
                r = torch.flatten(r, 1)
                r_c = torch.mean(r, 0)
                c_r_list.append(r_c)
            # c_r = torch.stack(c_r_list)
            task_r_dic[task_id] = c_r_list

        current_c_r_list = []
        classes = list(set(data.targets))
        for c in classes:
            c_index = [i for i, x in enumerate(data.targets) if x is c]
            c_data = data[c_index]
            x = c_data[0].cuda()
            t = c_data[-1].cuda()
            r = fea_extra.forward(x, t)
            r = torch.flatten(r, 1)
            r_c = torch.mean(r, 0)
            current_c_r_list.append(r_c)
        # current_c_r = torch.stack(current_c_r_list)

        cos = torch.nn.CosineSimilarity(0)
        dis_sim = torch.nn.PairwiseDistance(p=2)

        similarity_dic = {}
        for k, v in task_r_dic.items():
            sim_res = [dis_sim(torch.unsqueeze(i, 0), torch.unsqueeze(j, 0)) for i in current_c_r_list for j in v]
            dis = sum(sim_res)/len(sim_res)
            similarity_dic[k] = dis

        weights_dic = {}
        if len(similarity_dic) >= 2:
            d_order = sorted(similarity_dic.items(), key=lambda x: x[1], reverse=False)  # 升序排列，越小说明越不相似，需要给更大的 weight 值
            task_sim_order = [i[0] for i in d_order]
            return task_sim_order
    else:
        return [0]
