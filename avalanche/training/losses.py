import copy

import torch
from avalanche.training.plugins import SupervisedPlugin
from torch.nn import BCELoss
import numpy as np
from torch.autograd import Variable


class ICaRLLossPlugin(SupervisedPlugin):
    """
    ICaRLLossPlugin
    Similar to the Knowledge Distillation Loss. Works as follows:
        The target is constructed by taking the one-hot vector target for the
        current sample and assigning to the position corresponding to the
        past classes the output of the old model on the current sample.
        Doesn't work if classes observed in previous experiences might be
        observed again in future training experiences.
    """

    def __init__(self):
        super().__init__()
        self.criterion = BCELoss()

        self.old_classes = []
        self.old_model = None
        self.old_logits = None

    def before_forward(self, strategy, **kwargs):
        if self.old_model is not None:
            with torch.no_grad():
                self.old_logits = self.old_model(strategy.mb_x)

    def __call__(self, logits, targets):
        predictions = torch.sigmoid(logits)
        if len(logits.shape) != len(targets.shape):
            one_hot = torch.zeros(
                targets.shape[0],
                logits.shape[1],
                dtype=torch.float,
                device=logits.device,
            )

            one_hot[range(len(targets)), targets.long()] = 1
        else:
            one_hot = targets

        if self.old_logits is not None:
            old_predictions = torch.sigmoid(self.old_logits)
            one_hot[:, self.old_classes] = old_predictions[:, self.old_classes]
            self.old_logits = None

        return self.criterion(predictions, one_hot)

    def after_training_exp(self, strategy, **kwargs):
        if self.old_model is None:
            old_model = copy.deepcopy(strategy.model)
            old_model.eval()
            self.old_model = old_model.to(strategy.device)

        self.old_model.load_state_dict(strategy.model.state_dict())

        self.old_classes += np.unique(
            strategy.experience.dataset.targets
        ).tolist()


class ReplayMixupLossPlugin(SupervisedPlugin):
    def __init__(self):
        super().__init__()
        self.criterion = BCELoss()

        self.old_classes = []
        self.old_model = None
        self.old_logits = None

    def before_forward(self, strategy, **kwargs):
        if self.old_model is not None:
            with torch.no_grad():
                self.old_logits = self.old_model(strategy.mb_x)

    def __call__(self, logits, targets):
        if len(logits.shape) != len(targets.shape):
            one_hot = torch.zeros(
                targets.shape[0],
                logits.shape[1],
                dtype=torch.float,
                device=logits.device,
            )
            one_hot[range(len(targets)), targets.long()] = 1
        else:
            one_hot = targets
        predictions = torch.sigmoid(logits)

        if self.old_logits is not None:
            old_predictions = torch.sigmoid(self.old_logits)
            one_hot[:, self.old_classes] = old_predictions[:, self.old_classes]
            self.old_logits = None

        return self.criterion(predictions, one_hot)


class ParamsRegularizationLossPlugin(SupervisedPlugin):
    """
    ICaRLLossPlugin
    Similar to the Knowledge Distillation Loss. Works as follows:
        The target is constructed by taking the one-hot vector target for the
        current sample and assigning to the position corresponding to the
        past classes the output of the old model on the current sample.
        Doesn't work if classes observed in previous experiences might be
        observed again in future training experiences.
    """

    def __init__(self):
        super().__init__()
        self.criterion = BCELoss()

        self.current_model = None
        self.last_task_model = None
        self.auto_learn_w = {}


    def before_forward(self, strategy, **kwargs):
        self.current_model = strategy.model

    def __call__(self, logits, targets):
        return self.criterion(logits, targets) + self.regularization_loss()

    def after_training_exp(self, strategy, **kwargs):
        if self.last_task_model is None:
            self.last_task_model = copy.deepcopy(strategy.model)
            self.last_task_model.eval()

    def regularization_loss(self):
        regularization_loss = Variable(torch.FloatTensor(0),requires_grad=True)
        current_model = self.current_model
        for name,params in self.last_task_model.named_parameters():
            if 'bias' not in name:
                if name not in self.auto_learn_w.keys():
                    self.auto_learn_w[name] = Variable(torch.ones_like(params))
                regularization_loss += torch.mul(0.5, torch.pow(torch.sub(current_model[name], params), 2))



__all__ = ["ICaRLLossPlugin","ReplayMixupLossPlugin","ParamsRegularizationLossPlugin"]
