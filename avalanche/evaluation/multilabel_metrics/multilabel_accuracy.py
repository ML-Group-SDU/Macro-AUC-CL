################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 30-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import List, Union, Dict

import torch
from torch import Tensor
from avalanche.evaluation import Metric, PluginMetric, GenericPluginMetric
from avalanche.evaluation.multilabel_metrics.mean import Mean
from avalanche.evaluation.metric_utils import phase_and_task
from collections import defaultdict
from torcheval.metrics.functional import multiclass_f1_score
from torcheval.metrics.functional import multilabel_auprc
from torcheval.metrics import MultilabelAUPRC
from sklearn.metrics import roc_auc_score
import numpy as np

class Accuracy(Metric[float]):
    """Accuracy metric. This is a standalone metric.

    The update method computes the accuracy incrementally
    by keeping a running average of the <prediction, target> pairs
    of Tensors provided over time.

    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average accuracy
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an accuracy value of 0.
    """

    def __init__(self):
        self._mean_accuracy = Mean()

    @torch.no_grad()
    def update(
            self,
            predicted_y: Tensor,
            true_y: Tensor,
    ) -> None:
        """Update the running accuracy given the true and predicted labels.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.

        :return: None.
        """
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        true_y, predicted_y = eliminate_all_zero_columns(true_y, predicted_y)
        true_y, predicted_y = eliminate_all_one_columns(true_y, predicted_y)

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        # f1 = multiclass_f1_score(predicted_y, true_y, num_classes=81+81+21)
        # auprc = multilabel_auprc(predicted_y, true_y, average="macro")
        for i in range(true_y.shape[1]):
            if len(torch.unique(true_y[:, i])) != 2:
                print(true_y[:, i])
                raise ValueError(
                    "Only one class present in y_true. ROC AUC score "
                    "is not defined in that case."
                )

        if true_y.shape[1] > 0:
            true_y_numpy = true_y.cpu().numpy()
            predicted_y_numpy = predicted_y.cpu().numpy()

            roc_auc = roc_auc_score(true_y_numpy, predicted_y_numpy, average="macro")
            # roc_auc = roc_auc_score(true_y_numpy, predicted_y_numpy, average="micro", multi_class="ovr")

            total_patterns = len(true_y)
            self._mean_accuracy.update(
                roc_auc, total_patterns
            )


    def result(self) -> float:
        """Retrieves the running accuracy.

        Calling this method will not change the internal state of the metric.

        :return: The current running accuracy, which is a float value
            between 0 and 1.
        """
        return self._mean_accuracy.result()

    def reset(self) -> None:
        """Resets the metric.

        :return: None.
        """
        self._mean_accuracy.reset()


# class MultiLabelAccuracy(Accuracy):
#     def __init__(self):
#         self._mean_accuracy = Mean()
#
#
#     @torch.no_grad()
#     def update(
#         self,
#         predicted_y: Tensor,
#         true_y: Tensor,
#     ) -> None:
#         """Update the running accuracy given the true and predicted labels.
#
#         :param predicted_y: The model prediction. Both labels and logit vectors
#             are supported.
#         :param true_y: The ground truth. Both labels and one-hot vectors
#             are supported.
#
#         :return: None.
#         """
#         true_y = torch.as_tensor(true_y)
#         predicted_y = torch.as_tensor(predicted_y)
#
#         true_y,predicted_y = eliminate_all_zero_columns(true_y, predicted_y)
#         true_y,predicted_y = eliminate_all_one_columns(true_y,predicted_y)
#
#         if len(true_y) != len(predicted_y):
#             raise ValueError("Size mismatch for true_y and predicted_y tensors")
#
#         # f1 = multiclass_f1_score(predicted_y, true_y, num_classes=81+81+21)
#         # auprc = multilabel_auprc(predicted_y, true_y, average="macro")
#         for i in range(true_y.shape[1]):
#             if len(torch.unique(true_y[:,i])) != 2:
#                 print(true_y[:,i])
#                 raise ValueError(
#                     "Only one class present in y_true. ROC AUC score "
#                     "is not defined in that case."
#                 )
#
#         if true_y.shape[1] > 0:
#             true_y_numpy = true_y.cpu().numpy()
#             predicted_y_numpy = predicted_y.cpu().numpy()
#
#             roc_auc = roc_auc_score(true_y_numpy,predicted_y_numpy,average="macro",multi_class="ovo")
#
#             # acc = self.example_based_accuracy(predicted_y,true_y)
#
#             total_patterns = len(true_y)
#             self._mean_accuracy.update(
#                 roc_auc, total_patterns
#             )
#     def example_based_accuracy(self,
#                                predicted_y: Tensor,
#                                true_y: Tensor,
#                                ):
#         predicted_y_2 = torch.ge(predicted_y, 0.5) * 1
#         tp = torch.sum(torch.logical_and(true_y, predicted_y_2), axis=0)
#         fp = torch.sum(torch.logical_and(1 - true_y, predicted_y_2), axis=0)
#         tn = torch.sum(torch.logical_and(1 - true_y, 1 - predicted_y_2), axis=0)
#         fn = torch.sum(torch.logical_and(true_y, 1 - predicted_y_2), axis=0)
#         # precision = tp / (tp+fp)
#         # recall = tp / (tp+fn)
#         acc = (tp+tn)/(tp+tn+fp+fn)
#         # f1 = (tp*2)/(tp*2+fp+fn)
#         beta = 1.0
#         epsilon = 1e-5
#         macro_f1 = (1 + beta ** 2) * tp / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + epsilon)
#         # print(macro_f1)
#         res_mean = torch.nanmean(acc)
#         with open("/home/zhangyan/codes/avalanche/saves/f1.csv", "ab") as f:
#             d = macro_f1.cpu().numpy()[:21]
#             d = np.expand_dims(d,0)
#             np.savetxt(f,d,fmt="%.2f")
#         return torch.mean(macro_f1)

def eliminate_all_zero_columns(y_true,y_pred):
    indicators = torch.nonzero(torch.sum(y_true, dim=0))
    # indictors = torch.any(y_true.bool(),dim=0) # 如果有1，即有true，则返回的也是true。 所以False代表全是0

    if indicators.shape[0] > 0:
    # 需要消除全为0的列
        y_true_list = torch.split(y_true,1,1) # 将每个列单独拿出来
        y_pred_list = torch.split(y_pred,1,1)

        y_true_tmp = [i for num, i in enumerate(y_true_list) if num in indicators]
        y_pred_tmp = [i for num, i in enumerate(y_pred_list) if num in indicators]
        if indicators.shape[0] > 1:
            y_true,y_pred = torch.concat(y_true_tmp,1),torch.concat(y_pred_tmp,1)
        elif indicators.shape[0] == 1:
            y_true,y_pred = y_true_tmp[0],y_pred_tmp[0]
    elif indicators.shape[0] == 0:
        raise NotImplementedError

    return y_true,y_pred

def eliminate_all_one_columns(y_true,y_pred):
    retain_indexs = []
    for i in range(y_true.shape[1]):
        if len(torch.unique(y_true[:, i])) == 2:
            retain_indexs.append(i)
    if len(retain_indexs) == y_true.shape[1]:
        return y_true,y_pred
    else:
        a,b = y_true[:,retain_indexs],y_pred[:,retain_indexs]
    return a,b



class TaskAwareAccuracy(Metric[float]):
    """The task-aware Accuracy metric.

    The metric computes a dictionary of <task_label, accuracy value> pairs.
    update/result/reset methods are all task-aware.

    See :class:`avalanche.evaluation.Accuracy` for the documentation about
    the `Accuracy` metric.
    """

    def __init__(self):
        """Creates an instance of the task-aware Accuracy metric."""
        self._mean_accuracy = defaultdict(Accuracy)
        """
        The mean utility that will be used to store the running accuracy
        for each task label.
        """

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor,
        task_labels: Union[float, Tensor],
    ) -> None:
        """Update the running accuracy given the true and predicted labels.

        Parameter `task_labels` is used to decide how to update the inner
        dictionary: if Float, only the dictionary value related to that task
        is updated. If Tensor, all the dictionary elements belonging to the
        task labels will be updated.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param task_labels: the int task label associated to the current
            experience or the task labels vector showing the task label
            for each pattern.

        :return: None.
        """
        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        if isinstance(task_labels, Tensor) and len(task_labels) != len(true_y):
            raise ValueError("Size mismatch for true_y and task_labels tensors")

        if isinstance(task_labels, int):
            self._mean_accuracy[task_labels].update(predicted_y, true_y)
        elif isinstance(task_labels, Tensor):
            unique_task_labels = torch.unique(task_labels)
            for task_label in unique_task_labels:
                task_data_idxs = torch.where(task_labels == task_label)
                self._mean_accuracy[task_label.item()].update(
                    predicted_y[task_data_idxs[0],:], true_y[task_data_idxs[0],:]
                )
        else:
            raise ValueError(
                f"Task label type: {type(task_labels)}, "
                f"expected int/float or Tensor"
            )

    def result(self, task_label=None) -> Dict[int, float]:
        """
        Retrieves the running accuracy.

        Calling this method will not change the internal state of the metric.

        task label is ignored if `self.split_by_task=False`.

        :param task_label: if None, return the entire dictionary of accuracies
            for each task. Otherwise return the dictionary
            `{task_label: accuracy}`.
        :return: A dict of running accuracies for each task label,
            where each value is a float value between 0 and 1.
        """
        assert task_label is None or isinstance(task_label, int)

        if task_label is None:
            return {k: v.result() for k, v in self._mean_accuracy.items()}
        else:
            return {task_label: self._mean_accuracy[task_label].result()}

    def reset(self, task_label=None) -> None:
        """
        Resets the metric.
        task label is ignored if `self.split_by_task=False`.

        :param task_label: if None, reset the entire dictionary.
            Otherwise, reset the value associated to `task_label`.

        :return: None.
        """
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            self._mean_accuracy = defaultdict(Accuracy)
        else:
            self._mean_accuracy[task_label].reset()


class AccuracyPluginMetric(GenericPluginMetric[float]):
    """
    Base class for all accuracies plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        """Creates the Accuracy plugin

        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware accuracy or not.
        """
        self.split_by_task = split_by_task
        if self.split_by_task:
            self._accuracy = TaskAwareAccuracy()
        else:
            # self._accuracy = Accuracy()
            self._accuracy = Accuracy()

        super(AccuracyPluginMetric, self).__init__(
            self._accuracy, reset_at=reset_at, emit_at=emit_at, mode=mode
        )

    def reset(self, strategy=None) -> None:
        self._metric.reset()

    def result(self, strategy=None) -> float:
        return self._metric.result()

    def update(self, strategy):
        if isinstance(self._accuracy, Accuracy):

            self._accuracy.update(strategy.mb_output, strategy.mb_y)

        elif isinstance(self._accuracy, TaskAwareAccuracy):
            self._accuracy.update(
                strategy.mb_output, strategy.mb_y, strategy.mb_task_id
            )
        else:
            assert False, "should never get here."


class MinibatchAccuracy(AccuracyPluginMetric):
    """
    The minibatch plugin accuracy metric.
    This metric only works at training time.

    This metric computes the average accuracy over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochAccuracy` instead.
    """

    def __init__(self):
        """
        Creates an instance of the MinibatchAccuracy metric.
        """
        super(MinibatchAccuracy, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "Top1_MacroAUC_MB"


class EpochAccuracy(AccuracyPluginMetric):
    """
    The average accuracy over a single training epoch.
    This plugin metric only works at training time.

    The accuracy will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochAccuracy metric.
        """

        super(EpochAccuracy, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train"
        )

    def __str__(self):
        return "Top1_MacroAUC_Epoch"


class RunningEpochAccuracy(AccuracyPluginMetric):
    """
    The average accuracy across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the accuracy averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningEpochAccuracy metric.
        """

        super(RunningEpochAccuracy, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "Top1_RunningAcc_Epoch"


class ExperienceAccuracy(AccuracyPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average accuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceAccuracy, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "Top1_MacroAUC_Exp"


class StreamAccuracy(AccuracyPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average accuracy over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of StreamAccuracy metric
        """
        super(StreamAccuracy, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "Top1_MacroAUC_Stream"


class TrainedExperienceAccuracy(AccuracyPluginMetric):
    """
    At the end of each experience, this plugin metric reports the average
    accuracy for only the experiences that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of TrainedExperienceAccuracy metric by first
        constructing AccuracyPluginMetric
        """
        super(TrainedExperienceAccuracy, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )
        self._current_experience = 0

    def after_training_exp(self, strategy) -> None:
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience
        AccuracyPluginMetric.reset(self, strategy)
        return AccuracyPluginMetric.after_training_exp(self, strategy)

    def update(self, strategy):
        """
        Only update the accuracy with results from experiences that have been
        trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            AccuracyPluginMetric.update(self, strategy)

    def __str__(self):
        return "Accuracy_On_Trained_Experiences"


def macro_auc_metrics(
    *,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    stream=False,
    trained_experience=False,
) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch accuracy at training time.
    :param epoch: If True, will return a metric able to log
        the epoch accuracy at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch accuracy at training time.
    :param experience: If True, will return a metric able to log
        the accuracy on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the accuracy averaged over the entire evaluation stream of experiences.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation accuracy only for experiences that the
        model has been trained on

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchAccuracy())

    if epoch:
        metrics.append(EpochAccuracy())

    if epoch_running:
        metrics.append(RunningEpochAccuracy())

    if experience:
        metrics.append(ExperienceAccuracy())

    if stream:
        metrics.append(StreamAccuracy())

    if trained_experience:
        metrics.append(TrainedExperienceAccuracy())

    return metrics


__all__ = [
    "Accuracy",
    "TaskAwareAccuracy",
    "MinibatchAccuracy",
    "EpochAccuracy",
    "RunningEpochAccuracy",
    "ExperienceAccuracy",
    "StreamAccuracy",
    "TrainedExperienceAccuracy",
    "macro_auc_metrics",
]
