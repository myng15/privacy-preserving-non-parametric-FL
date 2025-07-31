"""Adapted from: https://github.com/omarfoq/knn-per/tree/main"""

from scipy.special import softmax
from sklearn.preprocessing import label_binarize
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class Learner:
    """
    Responsible of training and evaluating a (deep-)learning model

    Attributes
    ----------
    model (nn.Module): the model trained by the learner

    criterion (torch.nn.modules.loss): loss function used to train the `model`, should have reduction="none"

    metric (fn): function to compute the metric, should accept as input two vectors and return a scalar

    device (str or torch.device):

    optimizer (torch.optim.Optimizer):

    lr_scheduler (torch.optim.lr_scheduler):

    is_binary_classification (bool): whether to cast labels to float or not, if `BCELoss`
    is used as criterion this should be set to True

    Methods
    ------

    optimizer_step: perform one optimizer step, requires the gradients to be already computed.

    fit_batch: perform an optimizer step over one batch

    fit_batches: perform successive optimizer steps over successive batches

    evaluate_iterator: evaluate `model` on an iterator

    compute_embeddings_and_outputs: compute the embeddings and the outputs of all samples in an iterator

    get_param_tensor: get `model` parameters as a unique flattened tensor

    """
    def __init__(
            self,
            model,
            criterion,
            metric,
            device,
            optimizer,
            algorithm=None, 
            client_algorithm=None, 
            global_model=None, 
            model_name=None,
            lr_scheduler=None,
            is_binary_classification=False,
    ):
    
        self.global_model = global_model 
        self.mu = 0.05 # FedProx

        self.model = model.to(device)
        self.model_name = model_name
        self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        self.optimizer = optimizer
        self.algorithm = algorithm 
        self.client_algorithm = client_algorithm 
        self.lr_scheduler = lr_scheduler
            
        self.is_binary_classification = is_binary_classification

        self.is_ready = True

        self.n_modules = self.__get_num_modules()
        self.model_dim = int(self.get_param_tensor().shape[0])

    def __get_num_modules(self):
        """
        computes the number of modules in the model network;
        i.e., the size of `self.model.modules()`

        return:
            n_modules (int)
        """
        if not self.is_ready:
            return

        n_modules = 0
        for _ in self.model.modules():
            n_modules += 1

        return n_modules

    def compute_stochastic_gradient(self, batch, weights=None, frozen_modules=None):
        """
        compute the stochastic gradient on one batch, the result is stored in self.model.parameters()

        :param batch: (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :param frozen_modules: list of frozen modules; default is None

        :return:
            None
        """
        if frozen_modules is None:
            frozen_modules = []

        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        for frozen_module in frozen_modules:
            frozen_module.zero_grad()

        loss.backward()

    def fit_batch(self, batch, weights=None, frozen_modules=None):
        """
        perform an optimizer step over one batch drawn from `iterator`

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :param frozen_modules: list of frozen modules; default is None

        :return:
            loss.item()
            metric.item()

        """
        if frozen_modules is None:
            frozen_modules = []

        self.model.train()

        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)
        metric = self.metric(y_pred, y) / len(y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        for frozen_module in frozen_modules:
            frozen_module.zero_grad()

        loss.backward()

        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss.item(), metric.item()

    def fit_epoch(self, iterator, weights=None, frozen_modules=None):
        """
        perform several optimizer steps on all batches drawn from `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :param frozen_modules: list of frozen models; default is None

        :return:
            loss.item()
            metric.item()

        """
        if frozen_modules is None:
            frozen_modules = []

        self.model.train()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            n_samples += y.size(0)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            loss_vec = self.criterion(y_pred, y)
            
            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
            else:
                loss = loss_vec.mean()

            if self.algorithm == "fedprox":
                # Add FedProx regularization term
                self.global_model = self.global_model.to(self.device)
                prox_term = sum((param - global_param).norm(2) ** 2 for param, global_param in zip(self.model.parameters(), self.global_model.parameters()))
                loss += (self.mu) * prox_term 

            loss.backward()

            for frozen_module in frozen_modules:
                frozen_module.zero_grad()

            self.optimizer.step()

            if self.algorithm == "swad":
                self.client_algorithm.swad_step += 1
                self.client_algorithm.swad_model.update_parameters(self.model, step=self.client_algorithm.swad_step)

            global_loss += loss.item() * loss_vec.size(0)
            global_metric += self.metric(y_pred, y).item()

        return global_loss / n_samples, global_metric / n_samples

    def evaluate_iterator(self, iterator):
        """
        evaluate learner on `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader

        :return
            global_loss and  global_metric accumulated over the iterator

        """
        self.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        # For balanced accuracy computing
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for x, y, _ in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)

                y_pred = self.model(x)

                #AUC
                y_prob = torch.nn.functional.softmax(y_pred, dim=1)  

                global_loss += self.criterion(y_pred, y).sum().item()
                global_metric += self.metric(y_pred, y).item()

                _, predicted = torch.max(y_pred, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

                # Collect probabilities for ROC-AUC
                all_probs.extend(y_prob.cpu().numpy())

                n_samples += y.size(0)

        global_balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
        
        # Compute ROC-AUC score
        all_labels = np.array(all_labels)
        all_probs = np.vstack(all_probs)  

        unique_labels = np.unique(all_labels)
        if len(unique_labels) == 1:
            print("Warning: Only one class present in labels. Returning 0.5 (random performance) for ROC-AUC.")
            global_roc_auc = 0.5
        else:
            if len(unique_labels) == 2: 
                global_roc_auc = roc_auc_score(all_labels, all_probs[:, 1])

            else:
                # Mask irrelevant class logits before applying softmax
                mask = np.zeros_like(all_probs, dtype=bool)
                mask[:, unique_labels] = True
                masked_logits = np.where(mask, all_probs, -np.inf)
                probabilities = softmax(masked_logits, axis=1)  # Normalize valid classes
                
                global_roc_auc = roc_auc_score(all_labels, probabilities[:, unique_labels], multi_class='ovr', average='micro')

        return global_loss / n_samples, global_metric / n_samples, global_balanced_accuracy, global_roc_auc


    def fit_batches(self, iterator, n_steps, weights=None, frozen_modules=None):
        """
        perform successive optimizer steps over successive batches drawn from iterator

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_steps: number of successive batches
        :type n_steps: int
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :param frozen_modules: list of frozen models; default is None

        :return:
            average loss and metric over the `n_steps`

        """
        global_loss = 0
        global_acc = 0

        for step in range(n_steps):
            batch_loss, batch_acc = self.fit_batch(iterator, weights, frozen_modules=frozen_modules)
            global_loss += batch_loss
            global_acc += batch_acc

        return global_loss / n_steps, global_acc / n_steps

    def fit_epochs(self, iterator, n_epochs, weights=None, frozen_modules=None):
        """
        perform multiple training epochs

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_epochs: number of successive batches
        :type n_epochs: int
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :param frozen_modules: list of frozen models; default is None

        :return:
            None

        """
        for step in range(n_epochs):
            self.fit_epoch(iterator, weights, frozen_modules=frozen_modules)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def get_head(self):
        return list(self.model.children())[-1]

    def get_body(self):
        return list(self.model.children())[:-1]

    def freeze_body(self):
        body = self.get_body()

        for child in body:
            for param in child.parameters():
                param.requires_grad = False

    def unfreeze_body(self):
        body = self.get_body()

        for child in body:
            for param in child.parameters():
                param.requires_grad = False

    def get_param_tensor(self):
        """
        get `model` parameters as a unique flattened tensor

        :return: torch.tensor

        """
        param_list = []

        for param in self.model.parameters():
            param_list.append(param.data.view(-1, ))

        return torch.cat(param_list)

    def set_param_tensor(self, param_tensor):
        """
        sets the parameters of the model from `param_tensor`

        :param param_tensor: torch.tensor of shape (`self.model_dim`,)

        """
        param_tensor = param_tensor.to(self.device)

        current_index = 0
        for param in self.model.parameters():
            param_shape = param.data.shape
            current_dimension = param.data.view(-1, ).shape[0]

            param.data = \
                param_tensor[current_index: current_index + current_dimension].reshape(param_shape)

            current_index += current_dimension

    def get_grad_tensor(self):
        """
        get `model` gradients as a unique flattened tensor

        :return: torch.tensor

        """
        grad_list = []

        for param in self.model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.data.view(-1, ))

        return torch.cat(grad_list)

    def set_grad_tensor(self, grad_tensor):
        """

        :param grad_tensor: torch.tensor of shape (`self.model_dim`,)

        """
        grad_tensor = grad_tensor.to(self.device)

        current_index = 0
        for param in self.model.parameters():
            param_shape = param.data.shape
            current_dimension = param.data.view(-1, ).shape[0]

            param.grad.data = \
                deepcopy(grad_tensor[current_index: current_index + current_dimension].reshape(param_shape))

            current_index += current_dimension

    def free_gradients(self):
        """
        free memory allocated by gradients

        """

        self.optimizer.zero_grad(set_to_none=True)

    def free_memory(self):
        """
        free the memory allocated by the model weights

        """
        if not self.is_ready:
            return

        self.free_gradients()

        del self.lr_scheduler
        del self.optimizer
        del self.model

        self.is_ready = False

    def save_checkpoint(self, path):
        """
        save the model, the optimizer and thr learning rate scheduler state dictionaries

        :param: expected to be the path to a .pt file
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }

        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """
        load the model, the optimizer and thr learning rate scheduler state dictionaries

        :param: expected to be the path to a .pt file storing the required data
        """
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])