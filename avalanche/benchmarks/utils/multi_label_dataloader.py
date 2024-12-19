from itertools import chain
from typing import Dict, Optional, Sequence, Union,List

import torch
from torch.utils.data import RandomSampler
                              # DistributedSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader

from avalanche.benchmarks.utils import make_classification_dataset

from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.collate_functions import \
    classification_collate_mbatches_fn, mixup_classification_collate_mbatches_fn, traditional_mixup
from avalanche.benchmarks.utils.collate_functions import detection_collate_fn \
    as _detection_collate_fn
from avalanche.benchmarks.utils.collate_functions import \
    detection_collate_mbatches_fn as _detection_collate_mbatches_fn

_default_collate_mbatches_fn = classification_collate_mbatches_fn

detection_collate_fn = _detection_collate_fn

detection_collate_mbatches_fn = _detection_collate_mbatches_fn


class MultiTaskDataLoader: # Ori replay dataloader
    """Custom data loader for rehearsal/replay strategies."""

    def __init__(
        self,
        data: List[AvalancheDataset],
        oversample_small_tasks: bool = False,
        batch_size: int = 32,
        task_balanced_dataloader: bool = False,
        distributed_sampling: bool = True,
        class_num: int = 10,
        scenario_mode: str = '',
        **kwargs
    ):
        """Custom data loader for rehearsal strategies.

        This dataloader iterates in parallel two datasets, the current `data`
        and the rehearsal `memory`, which are used to create mini-batches by
        concatenating their data together. Mini-batches from both of them are
        balanced using the task label (i.e. each mini-batch contains a balanced
        number of examples from all the tasks in the `data` and `memory`).

        The length of the loader is determined only by the current
        task data and is the same than what it would be when creating a
        data loader for this dataset.

        If `oversample_small_tasks == True` smaller tasks are oversampled to
        match the largest task.

        :param data: AvalancheDataset.
        :param memory: AvalancheDataset.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param batch_size: the size of the data batch. It must be greater
            than or equal to the number of tasks.
        :param batch_size_mem: the size of the memory batch. If
            `task_balanced_dataloader` is set to True, it must be greater than
            or equal to the number of tasks.
        :param task_balanced_dataloader: if true, buffer data loaders will be
            task-balanced, otherwise it creates a single data loader for the
            buffer samples.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """
        if "collate_mbatches" in kwargs:
            raise ValueError(
                "collate_mbatches is not needed anymore and it has been "
                "deprecated. Data loaders will use the collate function"
                "`data.collate_fn`."
            )

        self.class_num = class_num
        self.scenario_mode = scenario_mode

        self.data = data
        self.oversample_small_tasks = oversample_small_tasks
        self.task_balanced_dataloader = task_balanced_dataloader
        self.data_batch_sizes: Union[int, Dict[int, int]] = dict()
        self.memory_batch_sizes: Union[int, Dict[int, int]] = dict()
        self.distributed_sampling = distributed_sampling
        self.loader_kwargs = kwargs

        if "collate_fn" in kwargs:
            self.collate_fn = kwargs["collate_fn"]
        else:
            self.collate_fn = self.data.collate_fn

        # collate is done after we have all batches
        # so we set an empty collate for the internal dataloaders
        self.loader_kwargs["collate_fn"] = lambda x: x


        self.data_batch_sizes, _ = self._get_batch_sizes(
            data, batch_size, 0, task_balanced_dataloader
        )


        loaders_for_len_estimation = []

        if isinstance(self.data_batch_sizes, int):
            loaders_for_len_estimation.append(
                _make_data_loader(
                    data,
                    distributed_sampling,
                    kwargs,
                    self.data_batch_sizes,
                    force_no_workers=True,
                )[0]
            )
        else:
            # Task balanced
            for task_id in data.task_set:
                dataset = data.task_set[task_id]
                mb_sz = self.data_batch_sizes[task_id]

                loaders_for_len_estimation.append(
                    _make_data_loader(
                        dataset,
                        distributed_sampling,
                        kwargs,
                        mb_sz,
                        force_no_workers=True,
                    )[0]
                )

        self.max_len = max([len(d) for d in loaders_for_len_estimation])

    def __iter__(self):

        loader_data, sampler_data = self._create_loaders_and_samplers(
            self.data, self.data_batch_sizes
        )
        iter_data_dataloaders = {}

        for t in loader_data.keys():
            iter_data_dataloaders[t] = iter(loader_data[t])

        max_len = max([len(d) for d in loader_data.values()])

        try:
            for it in range(max_len):
                mb_curr, mb_memory = [], []
                MultiLabelReplayDataLoader._get_mini_batch_from_data_dict(
                    iter_data_dataloaders,
                    sampler_data,
                    loader_data,
                    self.oversample_small_tasks,
                    mb_curr,
                )

                yield self.collate_fn(mb_curr)
                # yield [self.collate_fn(mb) for mb in mb_curr]
        except StopIteration:
            return

    def __len__(self):
        return self.max_len

    @staticmethod
    def _get_mini_batch_from_data_dict(
        iter_dataloaders,
        iter_samplers,
        loaders_dict,
        oversample_small_tasks,
        mb_curr,
    ):
        # list() is necessary because we may remove keys from the
        # dictionary. This would break the generator.
        for t in list(iter_dataloaders.keys()):
            t_loader = iter_dataloaders[t]
            t_sampler = iter_samplers[t]
            try:
                tbatch = next(t_loader)
            except StopIteration:
                # StopIteration is thrown if dataset ends.
                # reinitialize data loader
                if oversample_small_tasks:
                    # reinitialize data loader
                    if isinstance(t_sampler, DistributedSampler):
                        # Manage shuffling in DistributedSampler
                        t_sampler.set_epoch(t_sampler.epoch + 1)

                    iter_dataloaders[t] = iter(loaders_dict[t])
                    tbatch = next(iter_dataloaders[t])
                else:
                    del iter_dataloaders[t]
                    del iter_samplers[t]
                    continue
            mb_curr.extend(tbatch)
            # mb_curr.append(tbatch)


    def _create_loaders_and_samplers(self, data, batch_sizes,no_worker=False):
        loaders = dict()
        samplers = dict()

        if isinstance(batch_sizes, int):
            loader, sampler = _make_data_loader(
                data,
                self.distributed_sampling,
                self.loader_kwargs,
                batch_sizes,
                no_worker
            )
            loaders[0] = loader
            samplers[0] = sampler
        else:
            for task_id in data.task_set:
                dataset = data.task_set[task_id]
                mb_sz = batch_sizes[task_id]

                loader, sampler = _make_data_loader(
                    dataset,
                    self.distributed_sampling,
                    self.loader_kwargs,
                    mb_sz,
                    no_worker
                )

                loaders[task_id] = loader
                samplers[task_id] = sampler

        return loaders, samplers

    @staticmethod
    def _get_batch_sizes(data_dict, single_exp_batch_size, remaining_example,
                         task_balanced_dataloader):
        batch_sizes = dict()
        if task_balanced_dataloader:
            for task_id in data_dict.task_set:
                current_batch_size = single_exp_batch_size
                if remaining_example > 0:
                    current_batch_size += 1
                    remaining_example -= 1
                batch_sizes[task_id] = current_batch_size
        else:
            # Current data is loaded without task balancing
            batch_sizes = single_exp_batch_size
        return batch_sizes, remaining_example

    @staticmethod
    def _get_weighted_batch_sizes(data_dict, batch_size_mem, remaining_example, mu=0.7, similarity_order=None):
        whole_batch_size = batch_size_mem
        batch_sizes = dict()
        tasks_num = len(data_dict.task_set)

        list1 = [round(mu ** i, 2) for i in range(1, tasks_num+1)]
        task_num_ratio = [a/sum(list1) for a in list1]
        each_task_num = [int(whole_batch_size*a) for a in task_num_ratio]
        each_task_num[-1] = whole_batch_size - sum(each_task_num[:-1])
        assert sum(each_task_num) == whole_batch_size

        # for task_id in data_dict.task_set:

        for i in range(len(similarity_order)):
            batch_sizes[similarity_order[i]] = each_task_num[i]

        return batch_sizes, remaining_example


class MultiLabelReplayDataLoader: # Ori replay dataloader
    """Custom data loader for rehearsal/replay strategies."""

    def __init__(
        self,
        data: AvalancheDataset,
        memory: Optional[AvalancheDataset] = None,
        oversample_small_tasks: bool = False,
        batch_size: int = 32,
        batch_size_mem: int = 32,
        task_balanced_dataloader: bool = False,
        distributed_sampling: bool = True,
        class_num: int = 10,
        scenario_mode: str = '',
        **kwargs
    ):
        """Custom data loader for rehearsal strategies.

        This dataloader iterates in parallel two datasets, the current `data`
        and the rehearsal `memory`, which are used to create mini-batches by
        concatenating their data together. Mini-batches from both of them are
        balanced using the task label (i.e. each mini-batch contains a balanced
        number of examples from all the tasks in the `data` and `memory`).

        The length of the loader is determined only by the current
        task data and is the same than what it would be when creating a
        data loader for this dataset.

        If `oversample_small_tasks == True` smaller tasks are oversampled to
        match the largest task.

        :param data: AvalancheDataset.
        :param memory: AvalancheDataset.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param batch_size: the size of the data batch. It must be greater
            than or equal to the number of tasks.
        :param batch_size_mem: the size of the memory batch. If
            `task_balanced_dataloader` is set to True, it must be greater than
            or equal to the number of tasks.
        :param task_balanced_dataloader: if true, buffer data loaders will be
            task-balanced, otherwise it creates a single data loader for the
            buffer samples.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """
        if "collate_mbatches" in kwargs:
            raise ValueError(
                "collate_mbatches is not needed anymore and it has been "
                "deprecated. Data loaders will use the collate function"
                "`data.collate_fn`."
            )

        self.class_num = class_num
        self.scenario_mode = scenario_mode

        self.data = data
        self.memory = memory
        self.oversample_small_tasks = oversample_small_tasks
        self.task_balanced_dataloader = task_balanced_dataloader
        self.data_batch_sizes: Union[int, Dict[int, int]] = dict()
        self.memory_batch_sizes: Union[int, Dict[int, int]] = dict()
        self.distributed_sampling = distributed_sampling
        self.loader_kwargs = kwargs

        if "collate_fn" in kwargs:
            self.collate_fn = kwargs["collate_fn"]
        else:
            self.collate_fn = self.data.collate_fn

        # collate is done after we have all batches
        # so we set an empty collate for the internal dataloaders
        self.loader_kwargs["collate_fn"] = lambda x: x

        if task_balanced_dataloader:
            num_keys = len(self.memory.targets_task_labels.uniques)
            assert batch_size_mem >= num_keys, (
                "Batch size must be greator or equal "
                "to the number of tasks in the memory "
                "and current data."
            )

        self.data_batch_sizes, _ = self._get_batch_sizes(
            data, batch_size, 0, False
        )

        # Create dataloader for memory items
        if task_balanced_dataloader:
            num_keys = len(self.memory.targets_task_labels.uniques)
            single_group_batch_size = batch_size_mem // num_keys
            remaining_example = batch_size_mem % num_keys
        else:
            single_group_batch_size = batch_size_mem
            remaining_example = 0

        self.memory_batch_sizes, _ = self._get_batch_sizes(
            memory,
            single_group_batch_size,
            remaining_example,
            task_balanced_dataloader,
        )
        print("memory batch sizes:", self.memory_batch_sizes)

        loaders_for_len_estimation = []

        if isinstance(self.data_batch_sizes, int):
            loaders_for_len_estimation.append(
                _make_data_loader(
                    data,
                    distributed_sampling,
                    kwargs,
                    self.data_batch_sizes,
                    force_no_workers=True,
                )[0]
            )
        else:
            # Task balanced
            for task_id in data.task_set:
                dataset = data.task_set[task_id]
                mb_sz = self.data_batch_sizes[task_id]

                loaders_for_len_estimation.append(
                    _make_data_loader(
                        dataset,
                        distributed_sampling,
                        kwargs,
                        mb_sz,
                        force_no_workers=True,
                    )[0]
                )

        self.max_len = max([len(d) for d in loaders_for_len_estimation])


    def __iter__(self):

        loader_data, sampler_data = self._create_loaders_and_samplers(
            self.data, self.data_batch_sizes
        )

        loader_memory, sampler_memory = self._create_loaders_and_samplers(
            self.memory, self.memory_batch_sizes,
            no_worker=True
        )

        iter_data_dataloaders = {}
        iter_buffer_dataloaders = {}

        for t in loader_data.keys():
            iter_data_dataloaders[t] = iter(loader_data[t])
        for t in loader_memory.keys():
            iter_buffer_dataloaders[t] = iter(loader_memory[t])

        max_len = max([len(d) for d in loader_data.values()])

        try:
            for it in range(max_len):
                mb_curr,mb_memory = [],[]
                MultiLabelReplayDataLoader._get_mini_batch_from_data_dict(
                    iter_data_dataloaders,
                    sampler_data,
                    loader_data,
                    self.oversample_small_tasks,
                    mb_curr,
                )
                MultiLabelReplayDataLoader._get_mini_batch_from_data_dict(
                    iter_buffer_dataloaders,
                    sampler_memory,
                    loader_memory,
                    self.oversample_small_tasks,
                    mb_curr,
                )
                # iidds = []
                # for mb in mb_curr:
                #     ttt =[]
                #     for elem in mb:
                #         ttt.append(elem[1])
                #     ttt = torch.stack(ttt)
                #     bbb = torch.sum(ttt,dim=0)
                #     iidds.append( torch.where(bbb > 0))

                yield self.collate_fn(mb_curr)
                # yield [self.collate_fn(mb) for mb in mb_curr]
        except StopIteration:
            return

    def __len__(self):
        return self.max_len

    @staticmethod
    def _get_mini_batch_from_data_dict(
        iter_dataloaders,
        iter_samplers,
        loaders_dict,
        oversample_small_tasks,
        mb_curr,
    ):
        # list() is necessary because we may remove keys from the
        # dictionary. This would break the generator.
        for t in list(iter_dataloaders.keys()):
            t_loader = iter_dataloaders[t]
            t_sampler = iter_samplers[t]
            try:
                tbatch = next(t_loader)
            except StopIteration:
                # StopIteration is thrown if dataset ends.
                # reinitialize data loader
                if oversample_small_tasks:
                    # reinitialize data loader
                    if isinstance(t_sampler, DistributedSampler):
                        # Manage shuffling in DistributedSampler
                        t_sampler.set_epoch(t_sampler.epoch + 1)

                    iter_dataloaders[t] = iter(loaders_dict[t])
                    tbatch = next(iter_dataloaders[t])
                else:
                    del iter_dataloaders[t]
                    del iter_samplers[t]
                    continue
            mb_curr.extend(tbatch)
            # mb_curr.append(tbatch)


    def _create_loaders_and_samplers(self, data, batch_sizes,no_worker=False):
        loaders = dict()
        samplers = dict()

        if isinstance(batch_sizes, int):
            loader, sampler = _make_data_loader(
                data,
                self.distributed_sampling,
                self.loader_kwargs,
                batch_sizes,
                no_worker
            )
            loaders[0] = loader
            samplers[0] = sampler
        else:
            for task_id in data.task_set:
                dataset = data.task_set[task_id]
                mb_sz = batch_sizes[task_id]

                loader, sampler = _make_data_loader(
                    dataset,
                    self.distributed_sampling,
                    self.loader_kwargs,
                    mb_sz,
                    no_worker
                )

                loaders[task_id] = loader
                samplers[task_id] = sampler

        return loaders, samplers

    @staticmethod
    def _get_batch_sizes(data_dict, single_exp_batch_size, remaining_example,
                         task_balanced_dataloader):
        batch_sizes = dict()
        if task_balanced_dataloader:
            for task_id in data_dict.task_set:
                current_batch_size = single_exp_batch_size
                if remaining_example > 0:
                    current_batch_size += 1
                    remaining_example -= 1
                batch_sizes[task_id] = current_batch_size
        else:
            # Current data is loaded without task balancing
            batch_sizes = single_exp_batch_size
        return batch_sizes, remaining_example

    @staticmethod
    def _get_weighted_batch_sizes(data_dict, batch_size_mem, remaining_example, mu=0.7, similarity_order=None):
        whole_batch_size = batch_size_mem
        batch_sizes = dict()
        tasks_num = len(data_dict.task_set)

        list1 = [round(mu ** i, 2) for i in range(1, tasks_num+1)]
        task_num_ratio = [a/sum(list1) for a in list1]
        each_task_num = [int(whole_batch_size*a) for a in task_num_ratio]
        each_task_num[-1] = whole_batch_size - sum(each_task_num[:-1])
        assert sum(each_task_num) == whole_batch_size

        # for task_id in data_dict.task_set:

        for i in range(len(similarity_order)):
            batch_sizes[similarity_order[i]] = each_task_num[i]

        return batch_sizes, remaining_example


def _make_data_loader(
    dataset,
    distributed_sampling,
    data_loader_args,
    batch_size,
    force_no_workers=False,
):
    data_loader_args = data_loader_args.copy()
    data_loader_args["drop_last"] = True

    collate_from_data_or_kwargs(dataset, data_loader_args)

    if force_no_workers:
        data_loader_args['num_workers'] = 0
        if 'persistent_workers' in data_loader_args:
            data_loader_args['persistent_workers'] = False


    if _DistributedHelper.is_distributed and distributed_sampling:
        sampler = DistributedSampler(
            dataset,
            shuffle=data_loader_args.pop("shuffle", False),
            drop_last=data_loader_args.pop("drop_last", False),
        )
        data_loader = DataLoader(
            dataset, sampler=sampler, batch_size=batch_size, **data_loader_args
        )
    else:
        sampler = None
        data_loader = DataLoader(
            dataset, batch_size=batch_size, **data_loader_args
        )
    return data_loader, sampler


def collate_from_data_or_kwargs(data, kwargs):
    if "collate_fn" in kwargs:
        return
    elif hasattr(data, "collate_fn"):
        kwargs["collate_fn"] = data.collate_fn

class __DistributedHelperPlaceholder:
    is_distributed = False
    world_size = 1
    rank = 0

_DistributedHelper = __DistributedHelperPlaceholder()

__all__ = [
    "detection_collate_fn",
    "detection_collate_mbatches_fn",
    "collate_from_data_or_kwargs",
    "MultiLabelReplayDataLoader",
    "MultiTaskDataLoader",
]