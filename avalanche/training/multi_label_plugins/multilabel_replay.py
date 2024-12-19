from typing import Optional, TYPE_CHECKING

from avalanche.benchmarks.utils import concat_classification_datasets
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.benchmarks.utils.multi_label_dataloader import MultiLabelReplayDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ReweightedReplayExperienceBalancedBuffer,
    WRSExperienceBalancedBuffer
)

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate

class MultiLabelReplayPlugin(SupervisedPlugin):
    """
    """

    def __init__(
        self,
        mem_size: int = 200,
        batch_size: int = None,
        batch_size_mem: int = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
        with_wrs:str = "no"
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
            # self.storage_policy = MultiLabelClassBalancedBuffer(
            #     max_size=self.mem_size, adaptive_size=True
            # )
            if with_wrs == "yes":
                self.storage_policy = WRSExperienceBalancedBuffer(
                    max_size=self.mem_size, adaptive_size=True
                )
            else:
                self.storage_policy = ReweightedReplayExperienceBalancedBuffer(
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
        drop_last: bool = False,
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

        """
            the beta item
        """
        for k,v in self.storage_policy.buffer_groups.items():
            scale_ratio = v.pos_nums[0] / v.ori_pos_nums[0]
            strategy.rs[k] = scale_ratio

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = strategy.train_mb_size

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = strategy.train_mb_size



        class_num = len(strategy.experience.classes_in_this_experience)

        print("Here is ReplayDataloader, class num is :  ", class_num)


        strategy.dataloader = MultiLabelReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            class_num=class_num,
            scenario_mode=strategy.scenario_mode,
        )

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        self.storage_policy.update(strategy, **kwargs)
