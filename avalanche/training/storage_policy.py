import copy
import random
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, TYPE_CHECKING

import torch
from numpy import inf
from torch import cat, Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
import time

from avalanche.benchmarks.utils import (
    make_classification_dataset,
    classification_subset,
    AvalancheDataset,
)
from avalanche.models import FeatureExtractorBackbone
from ..benchmarks.utils.utils import concat_datasets
from torchvision.transforms import transforms

if TYPE_CHECKING:
    from .templates import SupervisedTemplate


class ExemplarsBuffer(ABC):
    """ABC for rehearsal buffers to store exemplars.

    `self.buffer` is an AvalancheDataset of samples collected from the previous
    experiences. The buffer can be updated by calling `self.update(strategy)`.
    """

    def __init__(self, max_size: int):
        """Init.

        :param max_size: max number of input samples in the replay memory.
        """
        self.max_size = max_size
        """ Maximum size of the buffer. """
        self._buffer: AvalancheDataset = concat_datasets([])

    @property
    def buffer(self) -> AvalancheDataset:
        """Buffer of samples."""
        return self._buffer

    @buffer.setter
    def buffer(self, new_buffer: AvalancheDataset):
        self._buffer = new_buffer

    @abstractmethod
    def update(self, strategy: "SupervisedTemplate", **kwargs):
        """Update `self.buffer` using the `strategy` state.

        :param strategy:
        :param kwargs:
        :return:
        """
        ...

    @abstractmethod
    def resize(self, strategy: "SupervisedTemplate", new_size: int):
        """Update the maximum size of the buffer.

        :param strategy:
        :param new_size:
        :return:
        """
        ...


class ReservoirSamplingBuffer(ExemplarsBuffer):
    """Buffer updated with reservoir sampling."""

    def __init__(self, max_size: int,c_nums):
        """
        :param max_size:
        """
        # The algorithm follows
        # https://en.wikipedia.org/wiki/Reservoir_sampling
        # We sample a random uniform value in [0, 1] for each sample and
        # choose the `size` samples with higher values.
        # This is equivalent to a random selection of `size_samples`
        # from the entire stream.
        super().__init__(max_size)
        # INVARIANT: _buffer_weights is always sorted.
        self._buffer_weights = torch.zeros(0)

        self.c_nums = c_nums

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        """Update buffer."""
        self.update_from_dataset(strategy.experience.dataset)

    def update_from_dataset(self, new_data: AvalancheDataset):
        """Update the buffer using the given dataset.

        :param new_data:
        :return:
        """
        new_weights = torch.rand(len(new_data))

        cat_weights = torch.cat([new_weights, self._buffer_weights])
        cat_data = new_data.concat(self.buffer)
        sorted_weights, sorted_idxs = cat_weights.sort(descending=True)

        buffer_idxs = sorted_idxs[: self.max_size]
        self.buffer = cat_data.subset(buffer_idxs)
        self._buffer_weights = sorted_weights[: self.max_size]

    def resize(self, strategy, new_size):
        """Update the maximum size of the buffer."""
        self.max_size = new_size
        if len(self.buffer) <= self.max_size:
            return
        self.buffer = classification_subset(
            self.buffer, torch.arange(self.max_size)
        )
        self._buffer_weights = self._buffer_weights[: self.max_size]

class ReweightedRSBuffer(ExemplarsBuffer):
    """Buffer updated with reservoir sampling."""

    def __init__(self, max_size: int,c_nums):
        """
        :param max_size:
        """
        # The algorithm follows
        # https://en.wikipedia.org/wiki/Reservoir_sampling
        # We sample a random uniform value in [0, 1] for each sample and
        # choose the `size` samples with higher values.
        # This is equivalent to a random selection of `size_samples`
        # from the entire stream.
        super().__init__(max_size)
        # INVARIANT: _buffer_weights is always sorted.
        self._buffer_weights = torch.zeros(0)

        self.c_nums = c_nums
        self.pos_nums = None
        self.neg_nums = None
        self.ori_pos_nums = None
        self.ori_neg_nums = None
        self.classes = list(c_nums.keys())

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        """Update buffer."""
        self.update_from_dataset(strategy.experience.dataset)

    def update_from_dataset(self, new_data: AvalancheDataset,device):
        """Update the buffer using the given dataset.

        :param new_data:
        :return:
        """
        self.device = device

        self.ratios,targets,self.ori_pos_nums, self.ori_neg_nums = self.cal_ratio_from_dataset(new_data,device)
        new_weights = torch.rand(len(new_data))

        cat_weights = torch.cat([new_weights, self._buffer_weights])
        cat_data = new_data.concat(self.buffer)
        sorted_weights, sorted_idxs = cat_weights.sort(descending=True)

        buffer_idxs = sorted_idxs[: self.max_size]
        self.buffer = cat_data.subset(buffer_idxs)
        self._buffer_weights = sorted_weights[: self.max_size]

        self.cal_ratio_from_dataset(self.buffer, self.device)

    def cal_ratio_from_dataset(self,new_data,device):
        loader = torch.utils.data.DataLoader(new_data,batch_size=128,shuffle=True)
        for i,data in enumerate(loader):
            if i == 0:
                targets = data[1]
            else:
                targets = torch.concat((targets,data[1]),0)
        del loader

        # targets = torch.stack(targets,0) if len(targets) > 1 else targets
        targets = targets.to(device)
        targets = self.eliminate_other_class_columns(targets) #���ص���һ����άtensor
        self.pos_nums = torch.count_nonzero(targets,dim=0)
        self.neg_nums = torch.sub(targets.shape[0],self.pos_nums)
        ratios = torch.div(self.pos_nums,self.neg_nums)

        return ratios,targets,self.pos_nums,self.neg_nums

    def eliminate_other_class_columns(self,target_tensor):

        indicators = torch.tensor(self.classes).to(self.device)
        specfic_target = target_tensor[:,indicators]
        # ��Ҫ����ȫΪ0����
        return specfic_target

    def resize(self, strategy, new_size):
        """Update the maximum size of the buffer."""
        self.max_size = new_size
        if len(self.buffer) <= self.max_size:
            return
        self.buffer = classification_subset(
            self.buffer, torch.arange(self.max_size)
        )
        self.cal_ratio_from_dataset(self.buffer, self.device)
        self._buffer_weights = self._buffer_weights[: self.max_size]


class RetainRatioSamplingBuffer(ExemplarsBuffer):
    """Buffer updated with retaining positive and negative ratio."""

    def __init__(self, max_size: int,c_nums):
        """
        :param max_size:
        """
        # The algorithm follows
        # https://en.wikipedia.org/wiki/Reservoir_sampling
        # We sample a random uniform value in [0, 1] for each sample and
        # choose the `size` samples with higher values.
        # This is equivalent to a random selection of `size_samples`
        # from the entire stream.
        super().__init__(max_size)
        # INVARIANT: _buffer_weights is always sorted.
        self.device = None
        self.ratios = None
        self.pos_nums,self.neg_nums = None,None
        self.ori_pos_nums, self.ori_neg_nums = None,None
        self.c_nums = c_nums
        self.classes = list(c_nums.keys())

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        """Update buffer."""
        self.update_from_dataset(strategy.experience.dataset)

    def update_from_dataset(self, new_data: AvalancheDataset,device):
        """Update the buffer using the given dataset.

        :param new_data:
        :return:
        """
        self.device = device
        self.ratios,targets,self.ori_pos_nums, self.ori_neg_nums = self.cal_ratio_from_dataset(new_data,device)

        # store using greedy algorithm
        idxs = []
        idx_for_store = None
        data_indexs = list(range(len(new_data)))
        if self.max_size >= len(new_data):
            self.buffer = new_data
            self.pos_nums, self.neg_nums = self.ori_pos_nums, self.ori_neg_nums
            return
        # �ǵ�Ҫ�����ظ���Ԫ��
        while len(idxs) < self.max_size:
            sub_sample_size = 200
            if len(data_indexs) < sub_sample_size:
                sub_sample_size = len(data_indexs)-1
            sub_indexs = random.sample(data_indexs,sub_sample_size)
            for step,idx in enumerate(sub_indexs):
                idxs.append(idx)
                t_idxs = torch.tensor(idxs,dtype=torch.long)
                ratios_tmp = self.cal_ratio_from_arrays(targets[t_idxs,:],device)
                gap_tmp = torch.mean(torch.abs(self.ratios-ratios_tmp))
                if step == 0:
                    gap = gap_tmp
                    idx_for_store = idx
                else:
                    if gap_tmp < gap:
                        gap = gap_tmp
                        idx_for_store = idx
                idxs.remove(idx)
            idxs.append(idx_for_store)
            data_indexs.remove(idx_for_store)
        self.buffer = new_data.subset(idxs)

        # self.buffer.add_transforms(transform=transforms.Compose([
        #     # transforms.ToTensor(),
        #     # transforms.Resize([sample_resolution, sample_resolution]),
        #     transforms.RandomRotation(30, expand=False),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomAdjustSharpness(sharpness_factor=random.randrange(10, 40, 1) / 10.0, p=0.4),
        #     transforms.RandomAutocontrast(p=0.3),
        # ]))


    def cal_ratio_from_dataset(self,new_data,device):
        loader = torch.utils.data.DataLoader(new_data,batch_size=128,shuffle=True)
        for i,data in enumerate(loader):
            if i == 0:
                targets = data[1]
            else:
                targets = torch.concat((targets,data[1]),0)
        del loader

        # targets = torch.stack(targets,0) if len(targets) > 1 else targets
        targets = targets.to(device)
        targets = self.eliminate_other_class_columns(targets) #���ص���һ����άtensor
        self.pos_nums = torch.count_nonzero(targets,dim=0)
        self.neg_nums = torch.sub(targets.shape[0],self.pos_nums)
        ratios = torch.div(self.pos_nums,self.neg_nums)

        return ratios,targets,self.pos_nums,self.neg_nums


    def cal_ratio_from_arrays(self,arrays,device):
        arrays = arrays.to(device)
        pos_nums = torch.count_nonzero(arrays, dim=0)
        neg_nums = torch.sub(arrays.shape[0], pos_nums)
        ratios = torch.div(pos_nums, torch.add(neg_nums,0.01))
        self.pos_nums, self.neg_nums = pos_nums, neg_nums
        return ratios


    def eliminate_other_class_columns(self,target_tensor):

        indicators = torch.tensor(self.classes).to(self.device)
        specfic_target = target_tensor[:,indicators]
        # ��Ҫ����ȫΪ0����
        return specfic_target


    def resize(self, strategy, new_size):
        """Update the maximum size of the buffer."""
        self.max_size = new_size
        if len(self.buffer) <= self.max_size:
            return

        # ̰���㷨�Ƴ�����
        _,targets,_,_ = self.cal_ratio_from_dataset(self.buffer,self.device)
        data_idxs = list(range(len(self.buffer)))
        while len(data_idxs) > new_size:
            sub_sample_size = 200
            if len(data_idxs) < sub_sample_size:
                sub_sample_size = len(data_idxs) - 1
            sub_idxs = random.sample(data_idxs, sub_sample_size)
            for step,idx in enumerate(sub_idxs):
                data_idxs.remove(idx)
                t_idxs = torch.tensor(data_idxs, dtype=torch.long)
                ratios_tmp = self.cal_ratio_from_arrays(targets[t_idxs, :], self.device)
                gap_tmp = torch.mean(torch.abs(self.ratios - ratios_tmp))
                if step == 0:
                    gap = gap_tmp
                    idx_for_remove = idx
                else:
                    if gap_tmp < gap:
                        gap = gap_tmp
                        idx_for_remove = idx
                data_idxs.append(idx)

            data_idxs.remove(idx_for_remove)

        self.buffer = classification_subset(
            self.buffer, data_idxs
        )


class BalancedExemplarsBuffer(ExemplarsBuffer):
    """A buffer that stores exemplars for rehearsal in separate groups.

    The grouping allows to balance the data (by task, experience,
    classes..). In combination with balanced data loaders, it can be used
    to sample balanced mini-batches during training.

    `self.buffer_groups` is a dictionary that stores each group as a
    separate buffer. The buffers are updated by calling
    `self.update(strategy)`.
    """

    def __init__(
        self, max_size: int, adaptive_size: bool = True, total_num_groups=None
    ):
        """
        :param max_size: max number of input samples in the replay memory.
        :param adaptive_size: True if max_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param total_num_groups: If adaptive size is False, the fixed number
                                of groups to divide capacity over.
        """
        super().__init__(max_size)
        self.adaptive_size = adaptive_size
        self.total_num_groups = total_num_groups
        if not self.adaptive_size:
            assert self.total_num_groups > 0, (
                "You need to specify `total_num_groups` if "
                "`adaptive_size=True`."
            )
        else:
            assert self.total_num_groups is None, (
                "`total_num_groups` is not compatible with "
                "`adaptive_size=False`."
            )

        self.buffer_groups: Dict[int, ExemplarsBuffer] = {}
        """ Dictionary of buffers. """

    @property
    def buffer_datasets(self):
        """Return group buffers as a list of `AvalancheDataset`s."""
        return [g.buffer for g in self.buffer_groups.values()]

    def get_group_lengths(self, num_groups):
        """Compute groups lengths given the number of groups `num_groups`."""
        if self.adaptive_size:
            lengths = [self.max_size // num_groups for _ in range(num_groups)]
            # distribute remaining size among experiences.
            rem = self.max_size - sum(lengths)
            for i in range(rem):
                lengths[i] += 1
        else:
            lengths = [
                self.max_size // self.total_num_groups
                for _ in range(num_groups)
            ]
        return lengths

    @property
    def buffer(self):
        return concat_datasets([g.buffer for g in self.buffer_groups.values()])

    @buffer.setter
    def buffer(self, new_buffer):
        assert NotImplementedError(
            "Cannot set `self.buffer` for this class. "
            "You should modify `self.buffer_groups instead."
        )

    @abstractmethod
    def update(self, strategy: "SupervisedTemplate", **kwargs):
        """Update `self.buffer_groups` using the `strategy` state.

        :param strategy:
        :param kwargs:
        :return:
        """
        ...

    def resize(self, strategy, new_size):
        """Update the maximum size of the buffers."""
        self.max_size = new_size
        lens = self.get_group_lengths(len(self.buffer_groups))
        for ll, buffer in zip(lens, self.buffer_groups.values()):
            buffer.resize(strategy, ll)


class ExperienceBalancedBuffer(BalancedExemplarsBuffer):
    """Rehearsal buffer with samples balanced over experiences.

    The number of experiences can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed experiences so far.
    """

    def __init__(
        self, max_size: int, adaptive_size: bool = True, num_experiences=None
    ):
        """
        :param max_size: max number of total input samples in the replay
            memory.
        :param adaptive_size: True if mem_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param num_experiences: If adaptive size is False, the fixed number
                                of experiences to divide capacity over.
        """
        super().__init__(max_size, adaptive_size, num_experiences)

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        new_data = strategy.experience.dataset
        num_exps = strategy.clock.train_exp_counter + 1
        lens = self.get_group_lengths(num_exps)

        c_nums = strategy.experience.dataset._datasets[0].c_nums

        new_buffer = ReservoirSamplingBuffer(lens[-1],c_nums)
        new_buffer.update_from_dataset(new_data)
        self.buffer_groups[num_exps - 1] = new_buffer

        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(strategy, ll)


# û��RRSsample��ֻ����ͨ��samples,Ϊreweighted loss���õ�
class ReweightedReplayExperienceBalancedBuffer(BalancedExemplarsBuffer):
    """Rehearsal buffer with samples balanced over experiences.

    The number of experiences can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed experiences so far.
    """

    def __init__(
        self,
        max_size: int,
        adaptive_size: bool = True,
        num_experiences=None
    ):
        """
        :param max_size: max number of total input samples in the replay
            memory.
        :param adaptive_size: True if mem_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param num_experiences: If adaptive size is False, the fixed number
                                of experiences to divide capacity over.
        """
        super().__init__(max_size, adaptive_size, num_experiences)
        self.rs={}

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        new_data = strategy.experience.dataset
        num_exps = strategy.clock.train_exp_counter + 1
        lens = self.get_group_lengths(num_exps)

        # cur_classes = strategy.experience.dataset._datasets[0].cur_classes
        c_nums = strategy.experience.dataset._datasets[0].c_nums

        new_buffer = ReweightedRSBuffer(lens[-1],c_nums)
        new_buffer.update_from_dataset(new_data,strategy.device)
        self.buffer_groups[num_exps - 1] = new_buffer

        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(strategy, ll)


class WRSExperienceBalancedBuffer(BalancedExemplarsBuffer):

    def __init__(
        self,
        max_size: int,
        adaptive_size: bool = True,
        num_experiences=None
    ):
        """
        :param max_size: max number of total input samples in the replay
            memory.
        :param adaptive_size: True if mem_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param num_experiences: If adaptive size is False, the fixed number
                                of experiences to divide capacity over.
        """
        super().__init__(max_size, adaptive_size, num_experiences)
        self.rs={}

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        new_data = strategy.experience.dataset
        num_exps = strategy.clock.train_exp_counter + 1
        lens = self.get_group_lengths(num_exps)

        # cur_classes = strategy.experience.dataset._datasets[0].cur_classes
        c_nums = strategy.experience.dataset._datasets[0].c_nums

        new_buffer = RetainRatioSamplingBuffer(lens[-1],c_nums)
        new_buffer.update_from_dataset(new_data,strategy.device)
        self.buffer_groups[num_exps - 1] = new_buffer

        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(strategy, ll)

class BasePRSOneBuffer(ExemplarsBuffer):
    """Buffer updated with reservoir sampling."""

    def __init__(self, max_size: int):
        """
        :param max_size:
        """
        # The algorithm follows
        # https://en.wikipedia.org/wiki/Reservoir_sampling
        # We sample a random uniform value in [0, 1] for each sample and
        # choose the `size` samples with higher values.
        # This is equivalent to a random selection of `size_samples`
        # from the entire stream.
        super().__init__(max_size)
        # INVARIANT: _buffer_weights is always sorted.
        self._buffer_weights = torch.zeros(0)
        self.ratios = None
        self.ori_pos_nums, self.ori_neg_nums = None, None
        self.pos_nums,self.neg_nums = None,None

    def update(self, strategy: "SupervisedTemplate",idx, **kwargs):
        """Update buffer."""
        self.update_from_sample(strategy,idx)

    def update_from_sample(self,strategy,idx):
        if len(self.buffer) < self.max_size:
            self.buffer = self.buffer.concat(strategy.experience.dataset.subset([idx]))

    def resize(self, data):
        """Update the maximum size of the buffer."""
        if len(self.buffer) <= self.max_size:
            raise ValueError("Error in BasePRSOneBuffer.resize")
        new_data = data
        # ̰���㷨�Ƴ�����
        ratios, targets, self.ori_pos_nums, self.ori_neg_nums = self.cal_ratio_from_dataset(new_data, self.device)
        buffer_idxs = list(range(len(self.buffer)))


        if self.ratios == None:
            self.ratios = ratios
        else:
            self.ratios = self.ratios + ratios
        # ̰���㷨�洢
        data_indexs = list(range(len(new_data)))

        # �ǵ�Ҫ�����ظ���Ԫ��
        # ��������ÿһ�����ݣ�����һ�£�������ӽ�ȥ���᲻���и��õı���������У���ӽ�ȥ�����û�У������
        for step, idx in enumerate(data_indexs):
            ori_ratios,_,_,_ = self.cal_ratio_from_dataset(self.buffer, self.device)
            gap_ori = torch.sum(torch.abs(self.ratios-ori_ratios)) # ԭ�ȵ�gap
            for step2,buffer_idx in enumerate(buffer_idxs):
                copy_idxs = copy.deepcopy(buffer_idxs)
                copy_idxs.remove(buffer_idx)
                buffer_sub = self.buffer.subset(copy_idxs)
                buffer_sub.concat(new_data.subset(idx))
                tmp_ratios,_,_,_ = self.cal_ratio_from_dataset(buffer_sub,self.device)
                gap_tmp = torch.sum((torch.abs(self.ratios-tmp_ratios))) # ���ڵ�gap
                if gap_tmp < gap_ori:
                    # �滻
                    self.buffer = buffer_sub
                    break
                else:
                    # ����ԭ״
                    pass


    def cal_ratio_from_dataset(self,new_data,device):
        loader = torch.utils.data.DataLoader(new_data,batch_size=128,shuffle=True)
        for i,data in enumerate(loader):
            if i == 0:
                targets = data[1]
            else:
                targets = torch.concat((targets,data[1]),0)
        del loader

        # targets = torch.stack(targets,0) if len(targets) > 1 else targets
        targets = targets.to(device)
        self.pos_nums = torch.count_nonzero(targets,dim=0)
        self.neg_nums = torch.sub(targets.shape[0],self.pos_nums) + 0.001
        ratios = torch.div(self.pos_nums,self.neg_nums)

        return ratios,targets,self.pos_nums,self.neg_nums

    def cal_ratio_from_arrays(self,arrays,device):
        arrays = arrays.to(device)
        pos_nums = torch.count_nonzero(arrays, dim=0)
        neg_nums = torch.sub(arrays.shape[0], pos_nums)
        ratios = torch.div(pos_nums, torch.add(neg_nums,0.01))
        self.pos_nums, self.neg_nums = pos_nums, neg_nums
        return ratios

class WRSWholeBuffer(BalancedExemplarsBuffer):
    """Buffer updated with retaining positive and negative ratio."""

    def __init__(
            self,
            max_size: int,
            adaptive_size: bool = True,
            total_num_classes: int = None,
    ):
        """Init.

        :param max_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        """
        if not adaptive_size:
            assert (
                    total_num_classes > 0
            ), """When fixed exp mem size, total_num_classes should be > 0."""

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.max_size = max_size
        self.total_num_classes = total_num_classes
        self.seen_classes = set()
        self.global_ratios = {}
        self.buffer_groups[0] = BasePRSOneBuffer(self.max_size)

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        c_nums = strategy.experience.dataset._datasets[0].c_nums
        self.device = strategy.device

        new_data = strategy.experience.dataset
        if len(self.buffer_groups[0].buffer) < self.max_size:
            for idx, target in enumerate(new_data.targets):
                if len(self.buffer_groups[0].buffer) < self.max_size:  # ���bufferû������ֱ�Ӵ��ȥ
                    self.save_sample(strategy, idx)
                else:  # ������ˣ�ɾһ������һ��
                    idxs_ = list(range(idx, len(new_data)))
                    self.resize(new_data.subset(idxs_))
                    break
        else:
            self.resize(new_data)
        self
        print(len(self.buffer))

    def resize(self, data):
        self.buffer_groups[0].resize(data)

    def save_sample(self,strategy,idx):
        self.buffer_groups[0].update(strategy,idx)


class BasePRSBuffer(ExemplarsBuffer):
    """Buffer updated with reservoir sampling."""

    def __init__(self, max_size: int):
        """
        :param max_size:
        """
        # The algorithm follows
        # https://en.wikipedia.org/wiki/Reservoir_sampling
        # We sample a random uniform value in [0, 1] for each sample and
        # choose the `size` samples with higher values.
        # This is equivalent to a random selection of `size_samples`
        # from the entire stream.
        super().__init__(max_size)
        # INVARIANT: _buffer_weights is always sorted.

    def update(self, strategy: "SupervisedTemplate",idx, **kwargs):
        """Update buffer."""
        self.update_from_sample(strategy,idx)

    def update_from_sample(self,strategy,idx):
        if len(self.buffer) < self.max_size:
            self.buffer = self.buffer.concat(strategy.experience.dataset.subset([idx]))

    def resize(self, strategy, new_size):
        pass

    def remove_sample(self,rsvr_idx):
        self.buffer = concat_datasets([self.buffer.subset(list(range(rsvr_idx))),self.buffer.subset(list(range(rsvr_idx,len(self.buffer))))])


class PartitionReservoirSamplingBuffer(BalancedExemplarsBuffer):
    """Buffer updated with retaining positive and negative ratio."""

    def __init__(
            self,
            max_size: int,
            adaptive_size: bool = True,
            total_num_classes: int = None,
    ):
        """Init.

        :param max_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        """
        if not adaptive_size:
            assert (
                    total_num_classes > 0
            ), """When fixed exp mem size, total_num_classes should be > 0."""

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.max_size = max_size
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes = set()

        self.buffer_groups[0] = BasePRSBuffer(self.max_size)

        self.p = {}
        self.n = {}

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        new_data = strategy.experience.dataset
        c_nums = new_data._datasets[0].c_nums
        self.device = strategy.device

        # ���� p
        self.current_p = {}
        ro = 0.1
        for k,v in c_nums.items():
            self.n[k.item()] = v[0]

        for k,v in self.n.items():
            n_ro = [torch.pow(e,ro) for e in self.n.values()]
            self.p[k] = torch.pow(v,ro)/torch.sum(torch.stack(n_ro))

        #
        idx = 0
        d = DataLoader(new_data,batch_size=128,num_workers=1)
        for batch_idx,samples in enumerate(d):
            targets = samples[1]
            for i in range(targets.shape[0]):
                target = targets[i,:]
                if len(self.buffer_groups[0].buffer) < self.max_size:
                    self.save_sample(strategy,idx)
                else:
                    if self.sample_in(target):
                        self.replace_sample(strategy,idx)
                idx += 1
        print(len(self.buffer_groups[0].buffer))

    def sample_in(self, targets):
        """
        determine sample can be in rsvr
        :param keys: substream names of sample
        :returns: True / False
        """
        keys = torch.where(targets != 0)[0]
        probs = [0. for _ in keys]
        negn = [0 for _ in keys]

        for i, key in enumerate(keys):
            mi = self.max_size * self.p[key.item()]
            ni = self.n[key.item()]
            # prob can't be larger than 1
            probs[i] = mi / ni if ni > mi else 1
            negn[i] = -ni

        probs = torch.FloatTensor(probs)
        weights = torch.FloatTensor(negn)
        weights = torch.softmax(weights, dim=0)

        s = torch.sum(probs * weights)

        return random.choices([True, False], [s, 1 - s])[0]

    def save_sample(self,strategy,idx):
        self.buffer_groups[0].update(strategy,idx)

    def replace_sample(self,strategy,idx):
        """
        replace given sample with old ones in rsvr.
        :param sample: sample to save in rsvr
        """
        rsvr_idx = self.sample_out()
        self.save_sample(strategy,idx)

    def sample_out(self):
        """
        evict a sample from rsvr
        :returns: removed sample idx of rsvr
        """
        targets = self.buffer_groups[0].buffer.targets
        a = [e for e in targets]
        targets = torch.stack([e for e in targets],0)
        buffer_c_nums = torch.sum(targets, 0)

        probs = {}
        # ֻ�ҵ�memory�漰�����
        for i,v in enumerate(buffer_c_nums):
            if v>0:
                if i in self.p.keys():
                    probs[i] = self.p[i]

        selected_key = random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]

        # y
        idxs = torch.where(targets[:,torch.tensor(selected_key)] == 1)[0]
        y = targets[idxs,:]

        l={}
        for c in probs.keys():
            l[c] = buffer_c_nums[c]
        deltas = {}
        for i in probs.keys():
            deltas[i] = l[i] - probs[i]*torch.sum(torch.stack(list(l.values())))

        # query
        query = torch.zeros(targets.shape[1])
        for key,delta in deltas.items():
            if delta <= 0:
                query[key] = 1

        # k
        scores = (1 - y) * query
        scores = scores.sum(axis=1)

        idxs_y =  torch.where(scores == torch.max(scores))[0]  # index within y
        k = [idxs[idx_y] for idx_y in idxs_y]

        def get_diff(targets_temp,probs):
            targets_temp = torch.stack(targets_temp,0)
            nums = torch.squeeze(torch.sum(targets_temp, 0))
            buffer_c_nums = {}
            for i,v in enumerate(nums):
                if v > 0:
                    buffer_c_nums[i] = nums[i]

            diffs=[]
            for k,v in buffer_c_nums.items():
                diffs.append(v-probs[k]*torch.sum(torch.stack(list(buffer_c_nums.values()))))
            diffs = torch.sum(torch.stack(diffs))
            return diffs
        # z
        best = (k[0], self.max_size)
        targets_temp = list(torch.split(targets, 1, 0))
        for idx in k:
            # remove sample
            idx = idx.item()
            a = targets_temp.pop(idx)
            diff = get_diff(targets_temp,probs)
            if diff < best[1]:
                best = (idx, diff)

            # save sample
            targets_temp.insert(idx,torch.unsqueeze(targets[idx,:],0))

        z = best[0]
        self.remove_sample(z)
        return z

    def remove_sample(self,idx):
        self.buffer_groups[0].remove_sample(idx)


class BaseOCDMBuffer(ExemplarsBuffer):
    """Buffer updated with reservoir sampling."""

    def __init__(self, max_size: int):
        """
        :param max_size:
        """
        # The algorithm follows
        # https://en.wikipedia.org/wiki/Reservoir_sampling
        # We sample a random uniform value in [0, 1] for each sample and
        # choose the `size` samples with higher values.
        # This is equivalent to a random selection of `size_samples`
        # from the entire stream.
        super().__init__(max_size)
        # INVARIANT: _buffer_weights is always sorted.
        self._buffer_weights = torch.zeros(0)

    def update(self, strategy: "SupervisedTemplate",idx, **kwargs):
        """Update buffer."""
        self.update_from_sample(strategy,idx)

    def update_from_sample(self,strategy,idx):
        if len(self.buffer) < self.max_size:
            self.buffer = self.buffer.concat(strategy.experience.dataset.subset([idx]))

    def resize(self, new_data, global_p):
        """Update the maximum size of the buffer."""
        cat_data = self.buffer.concat(new_data)
        targets = cat_data.targets
        targets = [e for e in targets]
        # targets = torch.split(targets,1,0)
        def cal_diff(m_p,global_p):
            assert m_p.keys() == global_p.keys()
            res = []
            for k in m_p.keys():
                res.append(torch.abs(torch.sub(m_p[k],global_p[k])))
            return torch.mean(torch.stack(res))
        def cal_p(targets:list,keys):
            targets = torch.stack(targets,0)
            nums = torch.sum(targets,0)
            c_nums = {}
            p = {}
            for c in keys:
                c_nums[c] = nums[c]
            for k, v in c_nums.items():
                n_ro = [torch.pow(e, 1) for e in c_nums.values()]
                p[k] = torch.pow(v, 1) / torch.sum(torch.stack(n_ro))
            return p

        idxs = list(range(len(targets)))
        retain_idxs = []
        while len(retain_idxs) < self.max_size:
            diff_tem = 100
            idx_tem = None
            sub_set = random.sample(idxs, 100)
            for idx in sub_set:
                retain_idxs.append(idx)
                sub_tar = [targets[i] for i in retain_idxs]
                m_p = cal_p(sub_tar, global_p.keys())
                diff = cal_diff(m_p, global_p)
                if diff < diff_tem:
                    diff_tem = diff
                    idx_tem = idx
                retain_idxs.remove(idx)
            retain_idxs.append(idx_tem)
            idxs.remove(idx_tem)
            print(len(retain_idxs))

        assert len(retain_idxs) == self.max_size
        self.buffer = cat_data.subset(retain_idxs)


class OCDMSamplingBuffer(BalancedExemplarsBuffer):
    """Buffer updated with retaining positive and negative ratio."""

    def __init__(
            self,
            max_size: int,
            adaptive_size: bool = True,
            total_num_classes: int = None,
    ):
        """Init.

        :param max_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        """
        if not adaptive_size:
            assert (
                    total_num_classes > 0
            ), """When fixed exp mem size, total_num_classes should be > 0."""

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.max_size = max_size
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes = set()

        self.buffer_groups[0] = BaseOCDMBuffer(self.max_size)

        self.p = {}
        self.n = {}

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        c_nums = strategy.experience.dataset._datasets[0].c_nums
        self.device = strategy.device

        # ���� p
        for k,v in c_nums.items():
            self.n[k.item()] = v[0]

        for k,v in self.n.items():
            n_ro = [torch.pow(e,0) for e in self.n.values()]
            self.p[k] = torch.pow(v,0)/torch.sum(torch.stack(n_ro))

        if len(self.buffer) < self.max_size:
            self.add_sample(strategy)
        else:
            self.resize(strategy.experience.dataset,self.p)
        print(len(self.buffer))

    def add_sample(self,strategy):
        new_data = strategy.experience.dataset
        for idx, target in enumerate(new_data.targets):
            if len(self.buffer_groups[0].buffer) < self.max_size:
                self.save_sample(strategy,idx)
            else:
                idxs_ = list(range(idx,len(new_data)))
                self.resize(new_data.subset(idxs_),self.p)
                break

    def resize(self, strategy, global_p):
        self.buffer_groups[0].resize(strategy,global_p)

    def save_sample(self,strategy,idx):
        self.buffer_groups[0].update(strategy,idx)


class ClassBalancedBuffer(BalancedExemplarsBuffer):
    """Stores samples for replay, equally divided over classes.

    There is a separate buffer updated by reservoir sampling for each class.
    It should be called in the 'after_training_exp' phase (see
    ExperienceBalancedStoragePolicy).
    The number of classes can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed classes so far.
    """

    def __init__(
        self,
        max_size: int,
        adaptive_size: bool = True,
        total_num_classes: int = None,
    ):
        """Init.

        :param max_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        """
        if not adaptive_size:
            assert (
                total_num_classes > 0
            ), """When fixed exp mem size, total_num_classes should be > 0."""

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes = set()

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        new_data = strategy.experience.dataset

        # Get sample idxs per class
        cl_idxs = {}
        for idx, target in enumerate(new_data.targets):
            if target not in cl_idxs:
                cl_idxs[target] = []
            cl_idxs[target].append(idx)

        # Make AvalancheSubset per class
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            cl_datasets[c] = classification_subset(new_data, indices=c_idxs)

        # Update seen classes
        self.seen_classes.update(cl_datasets.keys())

        # associate lengths to classes
        lens = self.get_group_lengths(len(self.seen_classes))
        class_to_len = {}
        for class_id, ll in zip(self.seen_classes, lens):
            class_to_len[class_id] = ll

        # update buffers with new data
        for class_id, new_data_c in cl_datasets.items():
            ll = class_to_len[class_id]
            if class_id in self.buffer_groups:
                old_buffer_c = self.buffer_groups[class_id]
                old_buffer_c.update_from_dataset(new_data_c)
                old_buffer_c.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(new_data_c)
                self.buffer_groups[class_id] = new_buffer

        # resize buffers
        for class_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[class_id].resize(
                strategy, class_to_len[class_id]
            )


class MultiLabelClassBalancedBuffer(BalancedExemplarsBuffer):
    """Stores samples for replay, equally divided over classes.

    There is a separate buffer updated by reservoir sampling for each class.
    It should be called in the 'after_training_exp' phase (see
    ExperienceBalancedStoragePolicy).
    The number of classes can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed classes so far.
    """

    def __init__(
        self,
        max_size: int,
        adaptive_size: bool = True,
        total_num_classes: int = None,
    ):
        """Init.

        :param max_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        """
        if not adaptive_size:
            assert (
                total_num_classes > 0
            ), """When fixed exp mem size, total_num_classes should be > 0."""

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes = set()

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        new_data = strategy.experience.dataset

        # Get sample idxs per class
        cl_idxs = {}
        for idx, target in enumerate(new_data.targets):
            target_indices = torch.where(target==1) # ��� multi_hot �� target
            for target in target_indices[0]:
                target = target.item()
                if target not in cl_idxs:
                    cl_idxs[target] = []
                cl_idxs[target].append(idx)

        # Make AvalancheSubset per class
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            cl_datasets[c] = classification_subset(new_data, indices=c_idxs)

        # Update seen classes
        self.seen_classes.update(cl_datasets.keys())

        # associate lengths to classes
        lens = self.get_group_lengths(len(self.seen_classes))
        class_to_len = {}
        for class_id, ll in zip(self.seen_classes, lens):
            class_to_len[class_id] = ll

        # update buffers with new data
        for class_id, new_data_c in cl_datasets.items():
            ll = class_to_len[class_id]
            if class_id in self.buffer_groups:
                old_buffer_c = self.buffer_groups[class_id]
                old_buffer_c.update_from_dataset(new_data_c)
                old_buffer_c.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(new_data_c)
                self.buffer_groups[class_id] = new_buffer

        # resize buffers
        for class_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[class_id].resize(
                strategy, class_to_len[class_id]
            )


class ParametricBuffer(BalancedExemplarsBuffer):
    """Stores samples for replay using a custom selection strategy and
    grouping."""

    def __init__(
        self,
        max_size: int,
        groupby=None,
        selection_strategy: Optional["ExemplarsSelectionStrategy"] = None,
    ):
        """Init.

        :param max_size: The max capacity of the replay memory.
        :param groupby: Grouping mechanism. One of {None, 'class', 'task',
            'experience'}.
        :param selection_strategy: The strategy used to select exemplars to
            keep in memory when cutting it off.
        """
        super().__init__(max_size)
        assert groupby in {None, "task", "class", "experience"}, (
            "Unknown grouping scheme. Must be one of {None, 'task', "
            "'class', 'experience'}"
        )
        self.groupby = groupby
        ss = selection_strategy or RandomExemplarsSelectionStrategy()
        self.selection_strategy = ss
        self.seen_groups = set()
        self._curr_strategy = None

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        new_data = strategy.experience.dataset
        new_groups = self._make_groups(strategy, new_data)
        self.seen_groups.update(new_groups.keys())

        # associate lengths to classes
        lens = self.get_group_lengths(len(self.seen_groups))
        group_to_len = {}
        for group_id, ll in zip(self.seen_groups, lens):
            group_to_len[group_id] = ll

        # update buffers with new data
        for group_id, new_data_g in new_groups.items():
            ll = group_to_len[group_id]
            if group_id in self.buffer_groups:
                old_buffer_g = self.buffer_groups[group_id]
                old_buffer_g.update_from_dataset(strategy, new_data_g)
                old_buffer_g.resize(strategy, ll)
            else:
                new_buffer = _ParametricSingleBuffer(
                    ll, self.selection_strategy
                )
                new_buffer.update_from_dataset(strategy, new_data_g)
                self.buffer_groups[group_id] = new_buffer

        # resize buffers
        for group_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[group_id].resize(
                strategy, group_to_len[group_id]
            )

    def _make_groups(self, strategy, data):
        """Split the data by group according to `self.groupby`."""
        if self.groupby is None:
            return {0: data}
        elif self.groupby == "task":
            return self._split_by_task(data)
        elif self.groupby == "experience":
            return self._split_by_experience(strategy, data)
        elif self.groupby == "class":
            return self._split_by_class(data)
        else:
            assert False, "Invalid groupby key. Should never get here."

    def _split_by_class(self, data):
        # Get sample idxs per class
        class_idxs = {}
        for idx, target in enumerate(data.targets):
            if target not in class_idxs:
                class_idxs[target] = []
            class_idxs[target].append(idx)

        # Make AvalancheSubset per class
        new_groups = {}
        for c, c_idxs in class_idxs.items():
            new_groups[c] = classification_subset(data, indices=c_idxs)
        return new_groups

    def _split_by_experience(self, strategy, data):
        exp_id = strategy.clock.train_exp_counter + 1
        return {exp_id: data}

    def _split_by_task(self, data):
        new_groups = {}
        for task_id in data.task_set:
            new_groups[task_id] = data.task_set[task_id]
        return new_groups


class _ParametricSingleBuffer(ExemplarsBuffer):
    """A buffer that stores samples for replay using a custom selection
    strategy.

    This is a private class. Use `ParametricBalancedBuffer` with
    `groupby=None` to get the same behavior.
    """

    def __init__(
        self,
        max_size: int,
        selection_strategy: Optional["ExemplarsSelectionStrategy"] = None,
    ):
        """
        :param max_size: The max capacity of the replay memory.
        :param selection_strategy: The strategy used to select exemplars to
                                   keep in memory when cutting it off.
        """
        super().__init__(max_size)
        ss = selection_strategy or RandomExemplarsSelectionStrategy()
        self.selection_strategy = ss
        self._curr_strategy = None

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        new_data = strategy.experience.dataset
        self.update_from_dataset(strategy, new_data)

    def update_from_dataset(self, strategy, new_data):
        self.buffer = self.buffer.concat(new_data)
        self.resize(strategy, self.max_size)

    def resize(self, strategy, new_size: int):
        self.max_size = new_size
        idxs = self.selection_strategy.make_sorted_indices(
            strategy=strategy, data=self.buffer
        )
        self.buffer = self.buffer.subset(idxs[: self.max_size])


class ExemplarsSelectionStrategy(ABC):
    """
    Base class to define how to select a subset of exemplars from a dataset.
    """

    @abstractmethod
    def make_sorted_indices(
        self, strategy: "SupervisedTemplate", data: AvalancheDataset
    ) -> List[int]:
        """
        Should return the sorted list of indices to keep as exemplars.

        The last indices will be the first to be removed when cutoff memory.
        """
        ...


class RandomExemplarsSelectionStrategy(ExemplarsSelectionStrategy):
    """Select the exemplars at random in the dataset"""

    def make_sorted_indices(
        self, strategy: "SupervisedTemplate", data: AvalancheDataset
    ) -> List[int]:
        indices = list(range(len(data)))
        random.shuffle(indices)
        return indices


class FeatureBasedExemplarsSelectionStrategy(ExemplarsSelectionStrategy, ABC):
    """Base class to select exemplars from their features"""

    def __init__(self, model: Module, layer_name: str):
        self.feature_extractor = FeatureExtractorBackbone(model, layer_name)

    @torch.no_grad()
    def make_sorted_indices(
        self, strategy: "SupervisedTemplate", data: AvalancheDataset
    ) -> List[int]:
        self.feature_extractor.eval()
        collate_fn = data.collate_fn if hasattr(data, "collate_fn") else None
        features = cat(
            [
                self.feature_extractor(x.to(strategy.device))
                for x, *_ in DataLoader(
                    data,
                    collate_fn=collate_fn,
                    batch_size=strategy.eval_mb_size,
                )
            ]
        )
        return self.make_sorted_indices_from_features(features)

    @abstractmethod
    def make_sorted_indices_from_features(self, features: Tensor) -> List[int]:
        """
        Should return the sorted list of indices to keep as exemplars.

        The last indices will be the first to be removed when cutoff memory.
        """


class HerdingSelectionStrategy(FeatureBasedExemplarsSelectionStrategy):
    """The herding strategy as described in iCaRL.

    It is a greedy algorithm, that select the remaining exemplar that get
    the center of already selected exemplars as close as possible as the
    center of all elements (in the feature space).
    """

    def make_sorted_indices_from_features(self, features: Tensor) -> List[int]:
        selected_indices = []

        center = features.mean(dim=0)
        current_center = center * 0

        for i in range(len(features)):
            # Compute distances with real center
            candidate_centers = current_center * i / (i + 1) + features / (
                i + 1
            )
            distances = pow(candidate_centers - center, 2).sum(dim=1)
            distances[selected_indices] = inf

            # Select best candidate
            new_index = distances.argmin().tolist()
            selected_indices.append(new_index)
            current_center = candidate_centers[new_index]

        return selected_indices


class ClosestToCenterSelectionStrategy(FeatureBasedExemplarsSelectionStrategy):
    """A greedy algorithm that selects the remaining exemplar that is the
    closest to the center of all elements (in feature space).
    """

    def make_sorted_indices_from_features(self, features: Tensor) -> List[int]:
        center = features.mean(dim=0)
        distances = pow(features - center, 2).sum(dim=1)
        return distances.argsort()

__all__ = [
    "ExemplarsBuffer",
    "ReservoirSamplingBuffer",
    "BalancedExemplarsBuffer",
    "ExperienceBalancedBuffer",
    "WRSExperienceBalancedBuffer",
    "ReweightedReplayExperienceBalancedBuffer",
    "ClassBalancedBuffer",
    "MultiLabelClassBalancedBuffer",
    "ParametricBuffer",
    "ExemplarsSelectionStrategy",
    "RandomExemplarsSelectionStrategy",
    "FeatureBasedExemplarsSelectionStrategy",
    "HerdingSelectionStrategy",
    "ClosestToCenterSelectionStrategy",
    "OCDMSamplingBuffer"
]
