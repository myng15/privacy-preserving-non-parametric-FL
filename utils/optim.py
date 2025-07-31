"""Adapted from: https://github.com/omarfoq/knn-per/tree/main"""

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required
import numpy as np
from copy import deepcopy


class FedOptimizer:
    def __init__(self, params, lr=1e-2, beta1=0.9, beta2=0.999, tau=1e-3, optimizer_name="fedadam"):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
        self.optimizer_name = optimizer_name

    def step(self, grads):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if self.optimizer_name == "fedadam":
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                param.data -= self.lr * m_hat / (v_hat.sqrt() + self.tau)
            elif self.optimizer_name == "fedadagrad":
                self.v[i] += grad ** 2
                param.data -= self.lr * grad / (self.v[i].sqrt() + self.tau)
            elif self.optimizer_name == "fedyogi":
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] -= (1 - self.beta2) * grad ** 2 * torch.sign(self.v[i] - grad ** 2)
                param.data -= self.lr * self.m[i] / (self.v[i].sqrt() + self.tau)



def get_optimizer(optimizer_name, model, lr_initial, mu=0., weight_decay=5e-4):
    """
    Gets torch.optim.Optimizer given an optimizer name, a model and learning rate

    :param optimizer_name: possible are adam and sgd
    :type optimizer_name: str
    :param model: model to be optimized
    :type optimizer_name: nn.Module
    :param lr_initial: initial learning used to build the optimizer
    :type lr_initial: float
    :param mu: proximal term weight; default=0.
    :type mu: float
    :return:
        torch.optim.Optimizer

    """

    if optimizer_name == "adam":
        return optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            weight_decay=weight_decay 
        )

    elif optimizer_name == "sgd":
        return optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            momentum=0.5, 
            weight_decay=weight_decay 
        )
    else:
        raise NotImplementedError("Other optimizer are not implemented")


def get_lr_scheduler(optimizer, scheduler_name, n_rounds=None, warmup_epochs=5): 
    """
    Gets torch.optim.lr_scheduler given an lr_scheduler name and an optimizer

    :param optimizer:
    :type optimizer: torch.optim.Optimizer
    :param scheduler_name: possible are
    :type scheduler_name: str
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`
    :type n_rounds: int
    :return: torch.optim.lr_scheduler

    """

    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    elif scheduler_name == "multi_step":
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
        milestones = [n_rounds//2, 3*(n_rounds//4)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    elif scheduler_name == "warmup":
        assert n_rounds is not None and warmup_epochs is not None, "Warmup scheduler requires local_epoch and warmup_epochs!"
        return optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=lambda ep: ((n_rounds * ep) / warmup_epochs) if ((n_rounds * ep) < warmup_epochs) else 1
        )
    else:
        raise NotImplementedError("Other learning rate schedulers are not implemented")
