import copy
from typing import Optional, Sequence, List, Union, Iterable

from torch.nn import Module, CrossEntropyLoss,BCELoss
from torch.optim import Optimizer, SGD

from avalanche.benchmarks import CLExperience
from avalanche.models.pnn import PNN
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import (
    SupervisedPlugin,
    EvaluationPlugin,
)
from avalanche.training.templates.base import ExpSequence
from avalanche.training.templates.multi_label_template import MultiLabelSupervisedTemplate,NaiveMarginSupervisedTemplate
from typing import Callable, Optional, Sequence, List, Union
from avalanche.benchmarks.utils.multi_label_dataloader import MultiTaskDataLoader
from pkg_resources import parse_version
import torch
from collections import defaultdict

from benchmarks.multi_label_dataset.common_utils import *
from avalanche.evaluation.multilabel_metrics.multiLabelLoss import ReweightedLoss,ReweightedMarginLoss,MarginLoss
from avalanche.training.multi_label_plugins.multilabel_replay import MultiLabelReplayPlugin
from avalanche.training.plugins.replay import ReplayPlugin




class MultiLabelNaive(MultiLabelSupervisedTemplate):
    """Naive finetuning.

    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.
    This strategy does not use task identities.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.dataloader
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
        do_initial=True,
        scale_ratio=1.0,
        **base_kwargs
    ):
        """
        Creates an instance of the Naive strategy.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )
        self.rs = {}
        if not do_initial:
            a = self.plugins
            for i in a:
                if i.__class__.__name__ == "PeriodicEval":
                    i.do_initial = False

        self.scale_ratio = scale_ratio


class MarginNaive(NaiveMarginSupervisedTemplate):
    """Naive finetuning.

    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.
    This strategy does not use task identities.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.dataloader
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
        do_initial=True,
        C=None,
        **base_kwargs
    ):
        """
        Creates an instance of the Naive strategy.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )
        self.C = C
        self.rs = {}
        if not do_initial:
            a = self.plugins
            for i in a:
                if i.__class__.__name__ == "PeriodicEval":
                    i.do_initial = False


class MultiTask(MultiLabelSupervisedTemplate):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
        do_initial=True,
        scale_ratio=1.0,
        **base_kwargs
    ):
        """
        Creates an instance of the Naive strategy.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )
        self.rs = {}
        if not do_initial:
            a = self.plugins
            for i in a:
                if i.__class__.__name__ == "PeriodicEval":
                    i.do_initial = False

        self.scale_ratio = scale_ratio


    def make_train_dataloader(
        self,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        persistent_workers=False,
        **kwargs
    ):

        other_dataloader_args = {"drop_last": True}

        if parse_version(torch.__version__) >= parse_version("1.7.0"):
            other_dataloader_args["persistent_workers"] = persistent_workers
        for k, v in kwargs.items():
            other_dataloader_args[k] = v
        try:
            total_class_num = self.experience[0].benchmark.n_classes
        except BaseException:
            total_class_num = -1
            print("no need for class num")

        print("make_train_dataloaders, numworkers", num_workers)
        print("total_class_num:", total_class_num)
        self.dataloader = MultiTaskDataLoader(
            self.adapted_dataset,
            num_workers=num_workers,
            oversample_small_tasks=True,
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            task_balanced_dataloader=True,
            class_num=total_class_num,
            **other_dataloader_args
        )


    def train(
        self,
        experiences: Union[CLExperience, ExpSequence],
        eval_streams: Optional[
            Sequence[Union[CLExperience, ExpSequence]]
        ] = None,
        **kwargs,
    ):
        """Training loop.

        If experiences is a single element trains on it.
        If it is a sequence, trains the model on each experience in order.
        This is different from joint training on the entire stream.
        It returns a dictionary with last recorded value for each metric.

        :param experiences: single Experience or sequence.
        :param eval_streams: sequence of streams for evaluation.
            If None: use training experiences for evaluation.
            Use [] if you do not want to evaluate during training.
            Experiences in `eval_streams` are grouped by stream name
            when calling `eval`. If you use multiple streams, they must
            have different names.
        """
        self.is_training = True
        self._stop_training = False

        self.model.train()
        self.model.to(self.device)
        from avalanche.benchmarks.utils.classification_dataset import concat_classification_datasets
        datasets = []
        for exp in experiences:
            datasets.append(exp.dataset)
        experiences = copy.deepcopy(experiences[0])
        experiences.dataset = concat_classification_datasets(datasets)

        # Normalize training and eval data.
        if not isinstance(experiences, Iterable):
            experiences = [experiences]
        if eval_streams is None:
            eval_streams = [experiences]


        self._eval_streams = _group_experiences_by_stream(eval_streams)

        self._before_training(**kwargs)

        for self.experience in experiences:
            self._before_training_exp(**kwargs)
            self._train_exp(self.experience, eval_streams, **kwargs)
            self._after_training_exp(**kwargs)
        self._after_training(**kwargs)


    def criterion(self):
        current_task_id = torch.tensor(self.experience.task_label)
        task_ids = torch.unique(self.mb_task_id)
        current_task_id = current_task_id.to(task_ids.device)
        try:
            assert current_task_id in task_ids
        except:
            raise ValueError(f"{current_task_id},{task_ids}",)

        if self.is_training:
            c_nums = {}
            data_size = {}
            for task_id, classification_dataset in enumerate(self.experience.dataset._datasets):
                c_nums[task_id] = classification_dataset._datasets[0].c_nums
                data_size[task_id] = len(classification_dataset)
        elif self.is_eval:
            c_nums = {}
            data_size = {}
            c_nums[self.experience.current_experience] = self.experience.dataset._datasets[0].c_nums
            data_size[self.experience.current_experience] = len(self.experience.dataset)

        losses = []
        for task_id in c_nums.keys():
            # calculate loss for current data
            indexs_t = torch.where(self.mb_task_id == task_id)
            if check_zeroorone(self.mb_y[indexs_t[0], :]):
                if isinstance(self._criterion, ReweightedLoss):
                    l = self._criterion(self.mb_output[indexs_t[0], :], self.mb_y[indexs_t[0], :], c_nums[task_id])
                elif isinstance(self._criterion, ReweightedMarginLoss) or isinstance(self._criterion, MarginLoss):
                    l = self._criterion(self.mb_output[indexs_t[0], :], self.mb_y[indexs_t[0], :], c_nums[task_id])
                else:
                    l = self._criterion(self.mb_output[indexs_t[0], :], self.mb_y[indexs_t[0], :])
                # losses.append(l)
                # print(torch.pow(torch.div(data_size[task_id], max(data_size.values())), 2))
                losses.append(torch.mul(l, torch.pow(torch.div(data_size[task_id], max(data_size.values())), 2)))

        if len(losses) == 1:
            return losses[0]
        else:
            return torch.sum(torch.stack(losses))



def _group_experiences_by_stream(eval_streams):
    if len(eval_streams) == 1:
        return eval_streams

    exps = []
    # First, we unpack the list of experiences.
    for exp in eval_streams:
        if isinstance(exp, Iterable):
            exps.extend(exp)
        else:
            exps.append(exp)
    # Then, we group them by stream.
    exps_by_stream = defaultdict(list)
    for exp in exps:
        sname = exp.origin_stream.name
        exps_by_stream[sname].append(exp)
    # Finally, we return a list of lists.
    return list(exps_by_stream.values())


def check_zeroorone(targets):
    for i in range(targets.shape[1]):
        cloumn = targets[:,i]
        if len(torch.unique(cloumn)) == 2:
            return True
    return False



__all__ = [
    "MultiLabelNaive",
    "MarginNaive",
]