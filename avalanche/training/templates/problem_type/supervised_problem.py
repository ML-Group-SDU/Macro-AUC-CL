import torch

from avalanche.models import avalanche_forward
from benchmarks.multi_label_dataset.common_utils import *
from avalanche.evaluation.multilabel_metrics.multiLabelLoss import ReweightedLoss,ReweightedMarginLoss,MarginLoss
from avalanche.training.multi_label_plugins.multilabel_replay import MultiLabelReplayPlugin
from avalanche.training.plugins.replay import ReplayPlugin

class SupervisedProblem:
    @property
    def mb_x(self):
        """Current mini-batch input."""
        return self.mbatch[0]

    @property
    def mb_y(self):
        """Current mini-batch target."""
        # return self.mbatch[1]

        # zy add here
        if not self.mixup:
            task_id = list(set(self.mb_task_id.tolist()))
            assert len(task_id) == 1
            return make_common_onehot(spec_onehot=self.mbatch[1], task_label=task_id[0])
        else: # perform mixup
            return self.mbatch[-2]


    @property
    def mb_task_id(self):
        """Current mini-batch task labels."""
        assert len(self.mbatch) >= 3
        return self.mbatch[-1]

    def criterion(self):
        """Loss function for supervised problems."""
        # print("loss function:", self._criterion)
        # c_nums = self.experience.dataset._datasets[0].c_nums

        return self._criterion(self.mb_output, self.mb_y)


    def forward(self):
        """Compute the model's output given the current mini-batch."""
        return avalanche_forward(self.model, self.mb_x, self.mb_task_id)

    def _check_minibatch(self):
        """Check if the current mini-batch has 3 components."""
        assert len(self.mbatch) >= 3


class MultiLabelSupervisedProblem:
    @property
    def mb_x(self):
        """Current mini-batch input."""
        return self.mbatch[0]

    @property
    def mb_y(self):
        """Current mini-batch target."""
        return self.mbatch[1]

    @property
    def mb_task_id(self):
        """Current mini-batch task labels."""
        assert len(self.mbatch) >= 3
        return self.mbatch[-1]

    def criterion(self):
        current_task_id = torch.tensor(self.experience.task_label)
        task_ids = torch.unique(self.mb_task_id)
        current_task_id = current_task_id.to(task_ids.device)
        assert current_task_id in task_ids
        c_nums = self.experience.dataset._datasets[0].c_nums


        losses = []
        # calculate loss for current data
        current_indexs = torch.where(self.mb_task_id==current_task_id)
        if check_zeroorone(self.mb_y[current_indexs[0],:]):
            if isinstance(self._criterion,ReweightedLoss):
                l1 = self._criterion(self.mb_output[current_indexs[0],:], self.mb_y[current_indexs[0],:],c_nums)

            elif isinstance(self._criterion,ReweightedMarginLoss) or isinstance(self._criterion,MarginLoss):
                l1 = self._criterion(self.mb_output[current_indexs[0],:], self.mb_y[current_indexs[0],:], c_nums)
            else:
                l1 = self._criterion(self.mb_output[current_indexs[0],:], self.mb_y[current_indexs[0],:])
            losses.append(l1)

        # calculate loss for memory data
        past_indexs = {}
        for task_id in task_ids:
            if task_id != current_task_id:
                mem_indexs = torch.where(self.mb_task_id==task_id)
                past_indexs[task_id] = mem_indexs

        if self.experience.task_label > 0 and self.is_eval is False:
            for k,ind in past_indexs.items():
                k = k.item()
                mb_y = self.mb_y[ind[0],:]
                if check_zeroorone(mb_y):
                    if isinstance(self._criterion,ReweightedLoss):
                        pl = self.get_plugin(MultiLabelReplayPlugin)
                        c_nums = pl.storage_policy.buffer_groups[k].c_nums
                        l = self._criterion(self.mb_output[ind[0],:], self.mb_y[ind[0],:],c_nums)
                    elif isinstance(self._criterion,ReweightedMarginLoss):
                        pl = self.get_plugin(MultiLabelReplayPlugin)
                        c_nums = pl.storage_policy.buffer_groups[k].c_nums
                        l = self._criterion(self.mb_output[ind[0], :], self.mb_y[ind[0], :], c_nums)
                    elif isinstance(self._criterion,MarginLoss):
                        pl = self.get_plugin(ReplayPlugin)
                        c_nums = pl.storage_policy.buffer_groups[k].c_nums
                        l = self._criterion(self.mb_output[ind[0], :], self.mb_y[ind[0], :], c_nums)
                    else: # BCE loss
                        l = self._criterion(self.mb_output[ind[0], :], self.mb_y[ind[0], :])

                    if l:
                        # the first item is for avoid overfitting
                        # the second item is beta, using for recover ratio
                        # losses.append(torch.mul(l, self.rs[k] * torch.tensor(0.6)))

                        size_kth_buffer = len(pl.storage_policy.buffer_groups[k].buffer)
                        size_current = len(self.experience.dataset)
                        rr = size_kth_buffer / size_current
                        losses.append(torch.mul(l, rr * self.scale_ratio))
                        # losses.append(l)

        if len(losses) == 1:
            return losses[0]
        else:
            return torch.sum(torch.stack(losses))

    def get_plugin(self, ins):
        for i in self.plugins:
            if isinstance(i, ins):
                return i
        return None

    def forward(self):
        """Compute the model's output given the current mini-batch."""
        return avalanche_forward(self.model, self.mb_x, self.mb_task_id)

    def _check_minibatch(self):
        """Check if the current mini-batch has 3 components."""
        assert len(self.mbatch) >= 3


class MarginNaiveSupervisedProblem:
    @property
    def mb_x(self):
        """Current mini-batch input."""
        return self.mbatch[0]

    @property
    def mb_y(self):
        """Current mini-batch target."""
        # return self.mbatch[1]

        # zy add here
        if not self.mixup:
            task_id = list(set(self.mb_task_id.tolist()))
            assert len(task_id) == 1
            return make_common_onehot(spec_onehot=self.mbatch[1], task_label=task_id[0])
        else: # perform mixup
            return self.mbatch[-2]


    @property
    def mb_task_id(self):
        """Current mini-batch task labels."""
        assert len(self.mbatch) >= 3
        return self.mbatch[-1]

    def criterion(self):
        current_task_id = torch.tensor(self.experience.task_label)
        task_ids = torch.unique(self.mb_task_id)
        assert current_task_id in task_ids
        c_nums = self.experience.dataset._datasets[0].c_nums


        losses = []
        # ��ǰ�����loss
        current_indexs = torch.where(self.mb_task_id==current_task_id)
        if check_zeroorone(self.mb_y[current_indexs[0],:]):

            if isinstance(self._criterion,MarginLoss):
                losses.append(
                    self._criterion(self.mb_output[current_indexs[0], :], self.mb_y[current_indexs[0], :], c_nums,self.C))
            else:
                losses.append(self._criterion(self.mb_output[current_indexs[0],:], self.mb_y[current_indexs[0],:]))

        # ������Ķ���memory��loss
        past_indexs = {}
        for task_id in task_ids:
            if task_id != current_task_id:
                mem_indexs = torch.where(self.mb_task_id==task_id)
                past_indexs[task_id] = mem_indexs

        if self.experience.task_label > 0 and self.is_eval is False:
            for k,ind in past_indexs.items():
                k = k.item()
                mb_y = self.mb_y[ind[0],:]
                if check_zeroorone(mb_y):
                    if isinstance(self._criterion,MarginLoss):
                        pl = None
                        for i in self.plugins:
                            if isinstance(i, ReplayPlugin):
                                pl = i
                        c_nums = pl.storage_policy.buffer_groups[k].c_nums
                        l = self._criterion(self.mb_output[ind[0], :], self.mb_y[ind[0], :], c_nums, self.C)
                    else:
                        l = self._criterion(self.mb_output[ind[0], :], self.mb_y[ind[0], :])

                    if l:
                        losses.append(l)

        if len(losses)==1:
            return losses[0]
        else:
            return torch.sum(torch.stack(losses))


    def forward(self):
        """Compute the model's output given the current mini-batch."""
        return avalanche_forward(self.model, self.mb_x, self.mb_task_id)

    def _check_minibatch(self):
        """Check if the current mini-batch has 3 components."""
        assert len(self.mbatch) >= 3


def check_zeroorone(targets):
    for i in range(targets.shape[1]):
        cloumn = targets[:,i]
        if len(torch.unique(cloumn)) == 2:
            return True
    return False