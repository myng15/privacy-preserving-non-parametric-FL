"""Adapted from: https://github.com/omarfoq/knn-per/tree/main"""

from copy import deepcopy
from opacus import GradSampleModule
import torch
import torch.nn as nn

import numpy as np

import warnings

from models.cgan import CGANTrainer
from models.swad import AveragedModel, LossValley


def average_learners(
        learners,
        target_learner,
        weights=None,
        average_params=True,
        average_gradients=False
):
    """
    Compute the average of a list of learner and store it into learner

    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learner, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :param average_params: if set to true the parameters are averaged; default is True
    :param average_gradients: if set to true the gradient are also averaged; default is False
    :type weights: torch.Tensor

    """
    if not average_params and not average_gradients:
        return

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)

    else:
        weights = weights.to(learners[0].device)

    target_state_dict = target_learner.model.state_dict(keep_vars=True)

    for key in target_state_dict:
        
        if target_state_dict[key].data.dtype == torch.float32:

            if average_params:
                target_state_dict[key].data.fill_(0.)

            if average_gradients:
                target_state_dict[key].grad = target_state_dict[key].data.clone()
                target_state_dict[key].grad.data.fill_(0.)

            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)

                if average_params:
                    target_state_dict[key].data += weights[learner_id] * state_dict[key].data.clone()

                if average_gradients:
                    if state_dict[key].grad is not None:
                        target_state_dict[key].grad += weights[learner_id] * state_dict[key].grad.clone()
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "trying to average_gradients before back propagation,"
                            " you should set `average_gradients=False`."
                        )

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()


def average_learners_fedadam(
        learners,
        target_learner,
        aggregator,
        weights=None,
        beta1=0.9,
        beta2=0.99, 
        tau=1e-8,
        lr=0.001
):
    """
    Compute the FedAdam update of a list of learners and store it in target_learner

    :param learners: List of clients' learners
    :param target_learner: The global model
    :param aggregator: CentralizedAggregator that stores FedAdam states
    :param weights: Weights for weighted averaging, if None, uniform averaging is used
    :param beta1: Momentum coefficient
    :param beta2: Adaptive second moment coefficient
    :param tau: Small value to prevent division by zero
    :param lr: Learning rate

    """

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)
    else:
        weights = weights.to(learners[0].device)

    assert torch.isclose(weights.sum(), torch.tensor(1.0, device=weights.device)), "Weights do not sum to 1!"

    target_state_dict = target_learner.model.state_dict(keep_vars=True)

    # Initialize if first round
    if not hasattr(aggregator, "m_t"):
        aggregator.m_t = {key: torch.zeros_like(target_state_dict[key].data) for key in target_state_dict}
        aggregator.v_t = {key: torch.zeros_like(target_state_dict[key].data) for key in target_state_dict}


    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:

            # Compute weighted average of gradients
            grad_avg = torch.zeros_like(target_state_dict[key].data)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)
                grad_avg += weights[learner_id] * state_dict[key].grad.clone()

            # FedAdam momentum update
            aggregator.m_t[key] = beta1 * aggregator.m_t[key] + (1 - beta1) * grad_avg
            aggregator.v_t[key] = beta2 * aggregator.v_t[key] + (1 - beta2) * (grad_avg ** 2)

            # Apply FedAdam update
            target_state_dict[key].data += lr * aggregator.m_t[key] / (torch.sqrt(aggregator.v_t[key]) + tau)


def average_learners_fedopt(
        learners,
        target_learner,
        aggregator,
        weights=None,
        beta1=0.9,
        beta2=0.99, 
        tau=1e-8,
        lr=0.001
):
    """
    Compute the FedAYogi update of a list of learners and store it in target_learner

    :param learners: List of clients' learners
    :param target_learner: The global model
    :param aggregator: CentralizedAggregator that stores FedAdam states
    :param weights: Weights for weighted averaging, if None, uniform averaging is used
    :param beta1: Momentum coefficient
    :param beta2: Adaptive second moment coefficient
    :param tau: Small value to prevent division by zero
    :param lr: Learning rate

    """

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)
    else:
        weights = weights.to(learners[0].device)
    
    assert torch.isclose(weights.sum(), torch.tensor(1.0, device=weights.device)), "Weights do not sum to 1!"

    target_state_dict = target_learner.model.state_dict(keep_vars=True)

    # Initialize if first round
    if not hasattr(aggregator, "m_t"):
        aggregator.m_t = {key: torch.zeros_like(target_state_dict[key].data) for key in target_state_dict}
        aggregator.v_t = {key: torch.zeros_like(target_state_dict[key].data) for key in target_state_dict}


    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:

            # Compute weighted average of gradients
            grad_avg = torch.zeros_like(target_state_dict[key].data)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)
                grad_avg += weights[learner_id] * state_dict[key].grad.clone()

            # General momentum update
            aggregator.m_t[key] = beta1 * aggregator.m_t[key] + (1 - beta1) * grad_avg

            if aggregator.algorithm == "fedadam":
                aggregator.v_t[key] = beta2 * aggregator.v_t[key] + (1 - beta2) * (grad_avg ** 2)
            
            elif aggregator.algorithm == "fedadagrad":
                aggregator.v_t[key] = aggregator.v_t[key] + (grad_avg ** 2)

            elif aggregator.algorithm == "fedyogi":
                aggregator.v_t[key] = aggregator.v_t[key] - (1 - beta2) * (grad_avg ** 2) * torch.sign(aggregator.v_t[key] - grad_avg ** 2)

            # Apply FedAdam update
            target_state_dict[key].data += lr * aggregator.m_t[key] / (torch.sqrt(aggregator.v_t[key]) + tau)


def average_trainers(
        learners,
        target_learner,
        weights=None,
        average_params=True,
        average_gradients=False
):
    """
    Compute the average of a list of CVAE-FEDAVG learner and store it into learner

    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learner, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :param average_params: if set to true the parameters are averaged; default is True
    :param average_gradients: if set to true the gradient are also averaged; default is False
    :type weights: torch.Tensor

    """
    if not average_params and not average_gradients:
        return
    
    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)

    else:
        weights = weights.to(learners[0].device)

    target_state_dict = target_learner.model._module.state_dict(keep_vars=True) if isinstance(target_learner.model, GradSampleModule) else target_learner.model.state_dict(keep_vars=True)

    decoder_keys = [key for key in target_state_dict if "decoder" in key]

    for key in decoder_keys:
        
        if target_state_dict[key].data.dtype == torch.float32:
            if average_params:
                target_state_dict[key].data.fill_(0.)

            if average_gradients:
                target_state_dict[key].grad = target_state_dict[key].data.clone()
                target_state_dict[key].grad.data.fill_(0.)

            for learner_id, learner in enumerate(learners):
                state_dict = learner.model._module.state_dict(keep_vars=True) if isinstance(learner.model, GradSampleModule) else learner.model.state_dict(keep_vars=True)
                
                if average_params:
                    target_state_dict[key].data += weights[learner_id] * state_dict[key].data.clone()

                if average_gradients:
                    if state_dict[key].grad is not None:
                        target_state_dict[key].grad += weights[learner_id] * state_dict[key].grad.clone()
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "trying to average_gradients before back propagation,"
                            " you should set `average_gradients=False`."
                        )

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model._module.state_dict() if isinstance(learner.model, GradSampleModule) else learner.model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()


def average_trainers_cgan(
        learners,
        target_learner,
        weights=None,
        average_params=True,
        average_gradients=False
):
    """
    Compute the average of a list of learner and store it into learner

    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learner, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :param average_params: if set to true the parameters are averaged; default is True
    :param average_gradients: if set to true the gradient are also averaged; default is False
    :type weights: torch.Tensor

    """
    if not average_params and not average_gradients:
        return
    
    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)

    else:
        weights = weights.to(learners[0].device)

    target_state_dict = target_learner.generator.state_dict(keep_vars=True) 

    for key in target_state_dict:
        
        if target_state_dict[key].data.dtype == torch.float32:
            if average_params:
                target_state_dict[key].data.fill_(0.)

            if average_gradients:
                target_state_dict[key].grad = target_state_dict[key].data.clone()
                target_state_dict[key].grad.data.fill_(0.)

            for learner_id, learner in enumerate(learners):
                state_dict = learner.generator.state_dict(keep_vars=True)
                
                if average_params:
                    target_state_dict[key].data += weights[learner_id] * state_dict[key].data.clone()

                if average_gradients:
                    if state_dict[key].grad is not None:
                        target_state_dict[key].grad += weights[learner_id] * state_dict[key].grad.clone()
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "trying to average_gradients before back propagation,"
                            " you should set `average_gradients=False`."
                        )

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.generator.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()
    

def copy_model(target, source):
    """
    Copy learners_weights from target to source
    :param target:
    :type target: nn.Module
    :param source:
    :type source: nn.Module
    :return: None

    """
    target.load_state_dict(source.state_dict())


def copy_encoder_only(target, source):
    target_state_dict = target._module.state_dict() if isinstance(target, GradSampleModule) else target.state_dict()
    source_state_dict = source._module.state_dict() if isinstance(source, GradSampleModule) else source.state_dict()

    for key in source_state_dict:
        if "encoder" in key:  # Ensure only encoder params are copied
            target_state_dict[key].copy_(source_state_dict[key])


def copy_decoder_only(target, source):
    target_state_dict = target._module.state_dict() if isinstance(target, GradSampleModule) else target.state_dict()
    source_state_dict = source._module.state_dict() if isinstance(source, GradSampleModule) else source.state_dict()
    
    for key in source_state_dict:
        if "decoder" in key:  # Ensure only decoder params are copied
            target_state_dict[key].copy_(source_state_dict[key])


class ClientAlgorithm:
    def __init__(self, model, device, is_swad=False):
        """
        Class to track data for each client.

        Args:
            is_swad (bool): Flag to indicate whether SWAD algorithm is enabled. Defaults to False.
        """

        self.is_swad = True 

        self.swad = None
        self.swad_early_stop = False
        self.device = device
        if is_swad:
            self.swad_n_converge = 3
            self.swad_n_tolerance = 6
            self.swad_tolerance_ratio = 0.05

            self.swad_step = 0
            self.swad = LossValley(
                n_converge=self.swad_n_converge,
                n_tolerance=self.swad_n_tolerance,
                tolerance_ratio=self.swad_tolerance_ratio,
            )
            self.swad_model = AveragedModel(model).to(device)
            self.swad_last_test_model = None

    def reset_SWAD(self, model):
        self.swad_early_stop = False
        self.swad = LossValley(
            n_converge=self.swad_n_converge,
            n_tolerance=self.swad_n_tolerance,
            tolerance_ratio=self.swad_tolerance_ratio,
        )
        self.swad_model = AveragedModel(model).to(self.device)
        self.swad_last_test_model = None
