import os
import pickle
import time
import random

from copy import deepcopy

from abc import ABC, abstractmethod

from opacus import GradSampleModule
import torch

from learners.learners_ensemble import LearnersEnsemble
from models.cgan import Discriminator
from models.cvae import CVAE
from utils.torch_utils import *

from tqdm import tqdm

import numpy as np
import numpy.linalg as LA

from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering

from faiss import IndexFlatL2, IndexIVFFlat

from collections import Counter, defaultdict

class Aggregator(ABC):
    r""" Base class for Aggregator. `Aggregator` dictates communications between clients
    Adapted from: https://github.com/omarfoq/knn-per/tree/main
    
    sampling_rate: proportion of clients used at each round; default is `1.`

    sample_with_replacement: is True, client are sampled with replacement; default is False

    c_round: index of the current communication round

    verbose: level of verbosity, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`

    rng: random number generator

    """
    def __init__(
            self,
            clients,
            global_learner,
            log_freq,
            global_train_logger,
            global_test_logger,
            algorithm=None,
            lr=None,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None
    ):

        rng_seed = seed 
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        self.global_learner = global_learner
        self.model_dim = self.global_learner.model_dim if self.global_learner else 0 
        self.device = self.global_learner.device if self.global_learner else None

        if test_clients is None:
            test_clients = []

        self.clients = clients
        self.test_clients = test_clients

        self.n_clients = len(clients)
        self.n_test_clients = len(test_clients)

        #Params for optimization algorithms (FedAdam, FedProx etc.)
        self.algorithm=algorithm
        self.lr = lr

        self.clients_weights =\
            torch.tensor(
                [client.n_train_samples for client in self.clients],
                dtype=torch.float32,
                device=self.device
            )

        self.clients_weights = self.clients_weights / self.clients_weights.sum()

        self.sampling_rate = sampling_rate
        self.sample_with_replacement = sample_with_replacement
        self.n_clients_per_round = max(1, int(self.sampling_rate * self.n_clients))
        self.sampled_clients_ids = list()
        self.sampled_clients = list()

        self.global_train_logger = global_train_logger
        self.global_test_logger = global_test_logger
        self.log_freq = log_freq
        self.verbose = verbose

        self.c_round = 0

    @abstractmethod
    def mix(self):
        pass

    @abstractmethod
    def toggle_client(self, client_id, mode):
        """
        toggle client at index `client_id`, if `mode=="train"`, `client_id` is selected in `self.clients`,
        otherwise it is selected in `self.test_clients`.

        :param client_id: (int)
        :param mode: possible are "train" and "test"
        """
        pass

    def toggle_clients(self):
        for client_id in range(self.n_clients):
            self.toggle_client(client_id, mode="train")

    def toggle_sampled_clients(self):
        for client_id in self.sampled_clients_ids:
            self.toggle_client(client_id, mode="train")

    def toggle_test_clients(self):
        for client_id in range(self.n_test_clients):
            self.toggle_client(client_id, mode="test")

    def write_logs(self, epoch=None, max_epochs=None, save_path=None):
        self.toggle_test_clients()

        for global_logger, clients, mode in [
            (self.global_train_logger, self.clients, "train"),
            (self.global_test_logger, self.test_clients, "test")
        ]:
            if len(clients) == 0:
                continue

            global_train_loss = 0.
            global_train_acc = 0.
            global_train_balanced_acc = 0.
            global_train_auc = 0.
            global_test_loss = 0.
            global_test_acc = 0.
            global_test_balanced_acc = 0.
            global_test_auc = 0.

            total_n_samples = 0
            total_n_test_samples = 0

            train_accuracies = []
            test_accuracies = []
            train_balanced_accuracies = []
            test_balanced_accuracies = []
            train_auc_scores = []
            test_auc_scores = []

            all_client_ids = []

            client_results_dict = {}

            for client in clients:
                train_loss, train_acc, train_balanced_acc, train_auc, test_loss, test_acc, test_balanced_acc, test_auc = client.write_logs()

                if self.verbose > 1:
                    print("*" * 30)
                    print(f"Client {client.id}..")
                    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train Balanced Acc: {train_balanced_acc * 100:.2f}% | Train ROC-AUC: {train_auc:.2f} |", end="")
                    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}% | Test Balanced Acc: {test_balanced_acc * 100:.2f}% | Test ROC-AUC: {test_auc:.2f} |")

                if save_path and epoch == max_epochs - 1:
                    results_dict = {
                        'client_id': client.id,
                        'epoch': epoch+1,
                        'train_loss': train_loss, 
                        'train_acc': train_acc * 100, 
                        'train_balanced_acc': train_balanced_acc * 100, 
                        'train_auc': train_auc,
                        'test_loss': test_loss, 
                        'test_acc': test_acc * 100, 
                        'test_balanced_acc': test_balanced_acc * 100,
                        'test_auc': test_auc, 
                        'n_train_samples': client.n_train_samples, 
                        'n_val_samples': client.n_val_samples, 
                        'n_test_samples': client.n_test_samples, 
                    } 
                    client_results_dict[client.id] = results_dict

                global_train_loss += train_loss * client.n_train_samples
                global_train_acc += train_acc * client.n_train_samples
                global_train_balanced_acc += train_balanced_acc * client.n_train_samples
                global_train_auc += train_auc * client.n_train_samples
                global_test_loss += test_loss * client.n_test_samples
                global_test_acc += test_acc * client.n_test_samples
                global_test_balanced_acc += test_balanced_acc * client.n_test_samples
                global_test_auc += test_auc * client.n_test_samples

                total_n_samples += client.n_train_samples
                total_n_test_samples += client.n_test_samples

                all_client_ids.append(client.id)
                train_accuracies.append(train_acc)
                test_accuracies.append(test_acc)
                train_balanced_accuracies.append(train_balanced_acc)
                test_balanced_accuracies.append(test_balanced_acc)
                train_auc_scores.append(train_auc)
                test_auc_scores.append(test_auc)

            global_train_loss /= total_n_samples
            global_test_loss /= total_n_test_samples
            global_train_acc /= total_n_samples
            global_train_balanced_acc /= total_n_samples
            global_test_acc /= total_n_test_samples
            global_test_balanced_acc /= total_n_test_samples

            # Calculate mean and standard deviations for train and test accuracies
            train_acc_mean, train_acc_std = np.mean(train_accuracies), np.std(train_accuracies)
            test_acc_mean, test_acc_std = np.mean(test_accuracies), np.std(test_accuracies)
            train_balanced_acc_mean, train_balanced_acc_std = np.mean(train_balanced_accuracies), np.std(train_balanced_accuracies)
            test_balanced_acc_mean, test_balanced_acc_std = np.mean(test_balanced_accuracies), np.std(test_balanced_accuracies)
            train_auc_mean, train_auc_std = np.mean(train_auc_scores), np.std(train_auc_scores)
            test_auc_mean, test_auc_std = np.mean(test_auc_scores), np.std(test_auc_scores)

            if self.verbose > 0:
                print("+" * 30)
                print("Global..")
                print(f"Train Loss: {global_train_loss:.3f} | Train Acc: {global_train_acc * 100:.2f}% | Train Balanced Acc: {global_train_balanced_acc * 100:.2f}% |", end="")
                print(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_acc * 100:.2f}% | Test Balanced Acc: {global_test_balanced_acc * 100:.2f}% |")
                print(f"Train Acc Mean/Std: {train_acc_mean * 100:.2f}%/{train_acc_std * 100:.2f}% | Test Acc Mean/Std: {test_acc_mean * 100:.2f}%/{test_acc_std * 100:.2f}% |")
                print(f"Train Balanced Acc Mean/Std: {train_balanced_acc_mean * 100:.2f}%/{train_balanced_acc_std * 100:.2f}% | Test Balanced Acc Mean/Std: {test_balanced_acc_mean * 100:.2f}%/{test_balanced_acc_std * 100:.2f}% |")
                print("+" * 50)

            global_logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
            global_logger.add_scalar("Train/Metric", global_train_acc, self.c_round)
            global_logger.add_scalar("Train/Balanced Acc", global_train_balanced_acc, self.c_round)
            global_logger.add_scalar("Train/ROC-AUC", global_train_auc, self.c_round)
            global_logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
            global_logger.add_scalar("Test/Metric", global_test_acc, self.c_round)
            global_logger.add_scalar("Test/Balanced Acc", global_test_balanced_acc, self.c_round)
            global_logger.add_scalar("Test/ROC-AUC", global_test_auc, self.c_round)

            if save_path and epoch == max_epochs - 1:
                global_results_dict = {
                    'all_client_ids': all_client_ids,
                    'epoch': epoch+1,
                    'global_train_loss': global_train_loss, 
                    'global_train_acc': global_train_acc * 100, 
                    'global_train_balanced_acc': global_train_balanced_acc * 100, 
                    'global_train_auc': global_train_auc,
                    'global_test_loss': global_test_loss, 
                    'global_test_acc': global_test_acc * 100, 
                    'global_test_balanced_acc': global_test_balanced_acc * 100, 
                    'global_test_auc': global_test_auc,
                    'total_n_samples': total_n_samples, 
                    'total_n_test_samples': total_n_test_samples, 
                }

                save_dir = os.path.join(save_path, mode)
                os.makedirs(save_dir, exist_ok=True)

                with open(os.path.join(save_dir, "global_results_dict.pkl"), 'wb') as f:
                    pickle.dump(global_results_dict, f)
                
                with open(os.path.join(save_dir, "client_results_dict.pkl"), 'wb') as f:
                    pickle.dump(client_results_dict, f)

        if self.verbose > 0:
            print("#" * 80)

    def evaluate(self):
        """
        evaluate the aggregator, returns the performance of every client in the aggregator

        :return
            clients_results: (np.array of size (self.n_clients, 2, 2))
                number of correct predictions and total number of samples per client both for train part and test part
            test_client_results: (np.array of size (self.n_test_clients))
                number of correct predictions and total number of samples per client both for train part and test part

        """

        clients_results = []
        test_client_results = []

        for results, clients, mode in [
            (clients_results, self.clients, "train"),
            (test_client_results, self.test_clients, "test")
        ]:
            if len(clients) == 0:
                continue

            print(f"evaluate {mode} clients..")
            for client_id, client in enumerate(tqdm(clients)):
                if not client.is_ready():
                    self.toggle_client(client_id, mode=mode)

                _, train_acc, train_balanced_acc, train_auc, _, test_acc, test_balanced_acc, test_auc = client.write_logs()

                results.append([
                    [train_acc * client.n_val_samples, client.n_val_samples],
                    [train_balanced_acc * client.n_train_samples, client.n_train_samples],
                    [train_auc * client.n_val_samples, client.n_val_samples],
                    [test_acc * client.n_test_samples, client.n_test_samples],
                    [test_balanced_acc * client.n_test_samples, client.n_test_samples],
                    [test_auc * client.n_test_samples, client.n_test_samples],
                ])

                if not isinstance(self, NoCommunicationAggregator):
                    client.free_memory()
                

        return np.array(clients_results, dtype=np.uint16), np.array(test_client_results, dtype=np.uint16)

    def save_state(self, dir_path):
        """
        save the state of the aggregator, i.e., the state dictionary of  `global_learner` as `.pt` file,
         and the state of each client in `self.clients`.

        :param dir_path:
        """
        save_path = os.path.join(dir_path, "global.pt")
        torch.save(self.global_learner.model.state_dict(), save_path)

        for client_id, client in enumerate(self.clients):
            self.toggle_client(client_id, mode="train")
            client.save_state()
            if not isinstance(self, NoCommunicationAggregator):
                client.free_memory()

    def load_state(self, dir_path):
        """
        load the state of the aggregator

        :param dir_path:
        """
        chkpts_path = os.path.join(dir_path, f"global.pt")
        self.global_learner.model.load_state_dict(torch.load(chkpts_path))
        for client_id, client in self.clients:
            self.toggle_client(client_id, mode="train")
            client.load_state()
            if not isinstance(self, NoCommunicationAggregator):
                client.free_memory()

    def sample_clients(self):
        """
        sample a list of clients without repetition
        """
        if self.sample_with_replacement:
            self.sampled_clients_ids = \
                self.rng.choices(
                    population=range(self.n_clients),
                    weights=self.clients_weights,
                    k=self.n_clients_per_round,
                )
        else:
            self.sampled_clients_ids = self.rng.sample(range(self.n_clients), k=self.n_clients_per_round)

        self.sampled_clients = [self.clients[id_] for id_ in self.sampled_clients_ids]


class CentralizedAggregator(Aggregator):
    r""" Standard Centralized Aggregator.
     All clients get fully synchronized with the average client.
    
    Adapted from: https://github.com/omarfoq/knn-per/tree/main
    """
    def mix(self):
        self.sample_clients()
        self.toggle_sampled_clients()

        for client in self.sampled_clients:
            client.learner.algorithm = self.algorithm

            #Debug - FedProx
            client.learner.global_model = deepcopy(self.global_learner.model)

            client.step()

        learners = [client.learner for client in self.sampled_clients]
            
        if self.algorithm in ["fedadam", "fedadagrad", "fedyogi"]:
            average_learners_fedopt(
                learners=learners,
                target_learner=self.global_learner,
                aggregator=self,
                weights=self.clients_weights[self.sampled_clients_ids] / self.sampling_rate,
                beta1=0.0 if self.algorithm == "fedadagrad" else 0.9,
                beta2=0.99,
                tau=1e-8,
                lr=self.lr 
            )
        else: # FedAvg
            average_learners(
                learners=learners,
                target_learner=self.global_learner,
                weights=self.clients_weights[self.sampled_clients_ids] / self.sampling_rate,
                average_params=True,
                average_gradients=False
            )

        for client in self.clients:
            copy_model(client.learner.model, self.global_learner.model)

        self.c_round += 1

    def toggle_client(self, client_id, mode):
        if mode == "train":
            client = self.clients[client_id]
        else:
            client = self.test_clients[client_id]

        if client.is_ready():
            copy_model(client.learner.model, self.global_learner.model)
        else:
            client.learner = deepcopy(self.global_learner)

        if callable(getattr(client.learner.optimizer, "set_initial_params", None)):
            client.learner.optimizer.set_initial_params(
                self.global_learner.model.parameters()
            )

    def save_state(self, dir_path):
        """
        save the state of the aggregator, i.e., the state dictionary of  `global_learner` as `.pt` file,
         and the state of each client in `self.clients`.

        :param dir_path:
        """
        save_path = os.path.join(dir_path, f"global_{self.c_round}.pt")
        torch.save(self.global_learner.model.state_dict(), save_path)

    def load_state(self, dir_path):
        """
        load the state of the aggregator

        :param dir_path:

        """
        chkpts_path = os.path.join(dir_path, f"global_{self.c_round}.pt")
        self.global_learner.model.load_state_dict(torch.load(chkpts_path))
 

class CVAEAggregator(Aggregator):
    r""" CVAE Centralized Aggregator.
     All clients get fully synchronized with the average client.

    """
    def __init__(
        self,
        clients,
        global_trainer,
        anonymizer,
        global_learner=None,
        log_freq=10,
        global_train_logger=None,
        global_test_logger=None,
        lr=None,
        verbose=1, 
        seed=None,
        *args, 
        **kwargs
    ):
        super().__init__(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            lr=lr,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=verbose,
            seed=seed,
            *args, 
            **kwargs
        )

        self.global_trainer = global_trainer
        self.device = self.global_trainer.device

        self.anonymizer = anonymizer

    def mix(self):
        self.sample_clients()
        self.toggle_sampled_clients()

        for client in self.sampled_clients:
            client.trainer = self.anonymizer.get_trainer(
                client=client, 
                global_model=self.global_trainer.model, 
                is_trained=True,
                num_epochs=5) 

        trainers = [client.trainer for client in self.sampled_clients]

        average_trainers(
            learners=trainers,
            target_learner=self.global_trainer,
            weights=self.clients_weights[self.sampled_clients_ids] / self.sampling_rate,
            average_params=True,
            average_gradients=False
        )

        for client in self.clients:
            copy_decoder_only(client.trainer.model, self.global_trainer.model)

        self.c_round += 1


    def toggle_client(self, client_id, mode):
        if mode == "train":
            client = self.clients[client_id]
        else:
            client = self.test_clients[client_id]

        if client.trainer.is_ready:
            copy_decoder_only(client.trainer.model, self.global_trainer.model)

        else:
            client.trainer = deepcopy(self.global_trainer)

        if callable(getattr(client.trainer.optimizer, "set_initial_params", None)):
            client.trainer.optimizer.set_initial_params(
                self.global_trainer.model.parameters()
            )

    def write_logs(self, anonymizer, epoch=None, max_epochs=None, save_path=None):
        global_val_loss = 0.
        total_n_val_samples = 0

        for client_id, client in enumerate(self.clients):
            _, val_loader = anonymizer.prepare_data(client.train_features, client.train_labels, client.val_features, client.val_labels)
            val_loss = client.trainer.evaluate(val_loader)

            if self.verbose > 0: 
                print("*" * 30)
                print(f"Client {client.id}..")
                print(f"Val Loss: {val_loss:.3f}")

            global_val_loss += val_loss * client.n_val_samples
            total_n_val_samples += client.n_val_samples

        global_val_loss /= total_n_val_samples

        if self.verbose > 0:
            print("+" * 30)
            print("Global..")
            print(f"Val Loss: {global_val_loss:.3f}")
            print("+" * 50)


    def save_state(self, dir_path):
        """
        save the state of the aggregator, i.e., the state dictionary of  `global_learner` as `.pt` file,
         and the state of each client in `self.clients`.

        :param dir_path:
        """
        global_ckpt_path = os.path.join(dir_path, "global.pt")

        checkpoint = {
            'model_state_dict': self.global_trainer.model._module.state_dict() if isinstance(self.global_trainer.model, GradSampleModule) else self.global_trainer.model.state_dict(), #self.global_trainer.model.state_dict(),
            'optimizer_state_dict': self.global_trainer.optimizer.state_dict()
        }

        if self.global_trainer.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.global_trainer.lr_scheduler.state_dict()

        torch.save(checkpoint, global_ckpt_path)
        print(f"Global model checkpoint saved successfully at {global_ckpt_path}.")


        for client in self.clients:
            self.toggle_client(client.id, mode="train")
            path = os.path.join(dir_path, f"client_{client.id}.pt")
            checkpoint = {
                'model_state_dict': client.trainer.model._module.state_dict() if isinstance(client.trainer.model, GradSampleModule) else client.trainer.model.state_dict(), #client.trainer.model.state_dict(),
                'optimizer_state_dict': client.trainer.optimizer.state_dict()
            }

            if client.trainer.lr_scheduler is not None:
                checkpoint['scheduler_state_dict'] = client.trainer.lr_scheduler.state_dict()

            torch.save(checkpoint, path)
            print(f"Client_{client.id}'s model checkpoint saved successfully at {path}.")


    def load_state(self, dir_path):
        """
        load the state of the aggregator

        :param dir_path:

        """
        chkpts_path = os.path.join(dir_path, f"global.pt")
        checkpoint = torch.load(chkpts_path, map_location=self.device, weights_only=True)

        for key in checkpoint["model_state_dict"]:
            if "decoder" in key:
                self.global_trainer.model.state_dict()[key].copy_(checkpoint["model_state_dict"][key])


        for client in self.clients:
            self.toggle_client(client.id, mode="train")
            path = os.path.join(dir_path, f"client_{client.id}.pt")
            checkpoint = torch.load(path)

            if isinstance(client.trainer.model, GradSampleModule):
                client.trainer.model._module.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(client.trainer.model, CVAE):
                client.trainer.model.load_state_dict(checkpoint["model_state_dict"])

            client.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint:
                client.trainer.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded successfully from {chkpts_path}.")



class CGANAggregator(Aggregator):
    r""" CGAN Centralized Aggregator.
     All clients get fully synchronized with the average client.

    """
    def __init__(
        self,
        clients,
        global_trainer,
        anonymizer,
        global_learner=None,
        log_freq=10,
        global_train_logger=None,
        global_test_logger=None,
        lr=None,
        verbose=1, 
        seed=None,
        *args, 
        **kwargs
    ):
        super().__init__(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            lr=lr,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=verbose,
            seed=seed,
            *args, 
            **kwargs
        )

        self.global_trainer = global_trainer
        self.device = self.global_trainer.device

        self.anonymizer = anonymizer

    def mix(self):
        self.sample_clients()
        self.toggle_sampled_clients()

        for client in self.sampled_clients:
            client.trainer = self.anonymizer.get_trainer(
                client=client, 
                global_model=self.global_trainer.generator, 
                is_trained=True,
                num_epochs=5) 

        trainers = [client.trainer for client in self.sampled_clients]

        average_trainers_cgan(
            learners=trainers,
            target_learner=self.global_trainer,
            weights=self.clients_weights[self.sampled_clients_ids] / self.sampling_rate,
            average_params=True,
            average_gradients=False
        )
        
        for client in self.clients:
            copy_model(client.trainer.generator, self.global_trainer.generator)

        self.c_round += 1


    def toggle_client(self, client_id, mode):
        if mode == "train":
            client = self.clients[client_id]
        else:
            client = self.test_clients[client_id]

        if client.trainer.is_ready:
            copy_model(client.trainer.generator, self.global_trainer.generator)

        else:
            client.trainer = deepcopy(self.global_trainer)

        if callable(getattr(client.trainer.g_optimizer, "set_initial_params", None)):
            client.trainer.g_optimizer.set_initial_params(
                self.global_trainer.generator.parameters()
            )

    def write_logs(self, anonymizer, epoch=None, max_epochs=None, save_path=None):
        global_val_g_loss = 0.
        global_val_d_accuracy = 0.
        total_n_val_samples = 0

        for client_id, client in enumerate(self.clients):
            _, val_loader = anonymizer.prepare_data(client.train_features, client.train_labels, client.val_features, client.val_labels)
            val_g_loss, val_d_accuracy = client.trainer.evaluate(val_loader)

            if self.verbose > 0: 
                print("*" * 30)
                print(f"Client {client.id}..")
                print(f"Val G_Loss: {val_g_loss:.3f} | Val D_Accuracy: {val_d_accuracy * 100:.3f}%")

            global_val_g_loss += val_g_loss * client.n_val_samples
            global_val_d_accuracy += val_d_accuracy * client.n_val_samples
            total_n_val_samples += client.n_val_samples

        global_val_g_loss /= total_n_val_samples
        global_val_d_accuracy /= total_n_val_samples

        if self.verbose > 0:
            print("+" * 30)
            print("Global..")
            print(f"Val G_Loss: {global_val_g_loss:.3f} | Val D_Accuracy: {global_val_d_accuracy * 100:.3f}%")
            print("+" * 50)


    def save_state(self, dir_path):
        """
        save the state of the aggregator, i.e., the state dictionary of  `global_learner` as `.pt` file,
         and the state of each client in `self.clients`.

        :param dir_path:
        """
        save_path = os.path.join(dir_path, "global.pt")

        checkpoint = {
            'generator_state_dict': self.global_trainer.generator.state_dict(), 
            'discriminator_state_dict': self.global_trainer.discriminator._module.state_dict() if isinstance(self.global_trainer.discriminator, GradSampleModule) else self.global_trainer.discriminator.state_dict(), 
            'g_optimizer_state_dict': self.global_trainer.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.global_trainer.d_optimizer.state_dict()
        }

        if self.global_trainer.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.global_trainer.lr_scheduler.state_dict()

        torch.save(checkpoint, save_path)


        for client in self.clients:
            self.toggle_client(client.id, mode="train")
            path = os.path.join(dir_path, f"client_{client.id}.pt")
            checkpoint = {
                'generator_state_dict': client.trainer.generator.state_dict(), 
                'discriminator_state_dict': client.trainer.discriminator._module.state_dict() if isinstance(client.trainer.discriminator, GradSampleModule) else client.trainer.discriminator.state_dict(), 
                'g_optimizer_state_dict': client.trainer.g_optimizer.state_dict(),
                'd_optimizer_state_dict': client.trainer.d_optimizer.state_dict()
            }

            if client.trainer.lr_scheduler is not None:
                checkpoint['scheduler_state_dict'] = client.trainer.lr_scheduler.state_dict()

            torch.save(checkpoint, path)

        print(f"Checkpoint saved successfully at {save_path}.")



    def load_state(self, dir_path):
        """
        load the state of the aggregator

        :param dir_path:

        """
        chkpts_path = os.path.join(dir_path, f"global.pt")
        checkpoint = torch.load(chkpts_path, map_location=self.device, weights_only=True)
        self.global_trainer.generator.load_state_dict(checkpoint["generator_state_dict"])

        if isinstance(self.global_trainer.discriminator, GradSampleModule):
            self.global_trainer.discriminator._module.load_state_dict(checkpoint['discriminator_state_dict'])
        elif isinstance(self.global_trainer.discriminator, Discriminator):
            self.global_trainer.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        
        for client in self.clients:
            self.toggle_client(client.id, mode="train")
            path = os.path.join(dir_path, f"client_{client.id}.pt")
            checkpoint = torch.load(path)

            client.trainer.generator.load_state_dict(checkpoint["generator_state_dict"])

            if isinstance(client.trainer.discriminator, GradSampleModule):
                client.trainer.discriminator._module.load_state_dict(checkpoint['discriminator_state_dict'])
            elif isinstance(client.trainer.discriminator, Discriminator):
                client.trainer.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

            client.trainer.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint:
                client.trainer.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded successfully from {chkpts_path}.")


class FedKNNAggregator:
    r""" Aggregator for Non-parametric (Centroid-based) Federated Data Sharing Methods.
     All clients get entire global datastore (containing anonymized data shared by clients) from server.

    """
    def __init__(self, num_clients, features_dimension, seed, max_global_features_per_client=1e6):
        self.num_clients = num_clients
        self.features_dimension = features_dimension
        self.seed = seed
        self.global_features = []
        self.global_labels = []
        self.max_global_features_per_client = max_global_features_per_client
        self.all_clients_indices = {} # Keeps track of which data the global aggregator receives from which client
        self.correct_indices = []

    def get_data_from_client(self, client_id, features, labels):
        start_idx = len(self.global_features)
        self.global_features.extend(features)
        self.global_labels.extend(labels)
        end_idx = len(self.global_features)
        self.all_clients_indices[client_id] = (start_idx, end_idx)
 
    def aggregate_all_clients_data(self):
        # Store features and labels received from all clients
        self.global_features = np.vstack(self.global_features)
        self.global_labels = np.array(self.global_labels)

        assert len(self.global_features) == len(self.global_labels), (
            f"Global feature-label mismatch: {len(self.global_features)} features, {len(self.global_labels)} labels"
        )

    # ----- FedProto: Prototype aggregation -----
    def aggregate_prototypes(self, local_proto_list):
        global_protos = defaultdict(list)
        for client_protos in local_proto_list:
            for c, proto in client_protos.items():
                global_protos[c].append(proto)
        for c in global_protos:
            global_protos[c] = np.mean(global_protos[c], axis=0)
        return global_protos

    def get_proto_data(self, global_protos, shuffle=True):
        # Stack all prototypes and labels
        prototypes = []
        labels = []
        for cls, proto in global_protos.items():
            prototypes.append(proto)
            labels.append(cls)

        prototypes = np.vstack(prototypes)  # shape: (num_embeddings, emb_dim)
        labels = np.array(labels)          # shape: (num_embeddings,)

        if shuffle:
            random.seed(self.seed)
            os.environ['PYTHONHASHSEED'] = str(self.seed)
            np.random.seed(self.seed)

            indices = np.random.permutation(len(labels))
            prototypes = prototypes[indices]
            labels = labels[indices]

        return prototypes, labels

    def clean_misclassified_samples(self, method="exact", n_neighbors=5, n_list=50):
        original_global_size = len(self.global_features)
        if len(self.global_features) == 0:
            print("No global data to clean.")
            return

        if method == "exact":
            # Initialize FAISS IndexFlatL2 (brute force)
            index = IndexFlatL2(self.features_dimension)
            index.add(self.global_features)
        elif method == "approximate":
            # Initialize FAISS IndexIVFFlat
            quantizer = IndexFlatL2(self.features_dimension)  # Coarse quantizer
            index = IndexIVFFlat(quantizer, self.features_dimension, n_list)
            index.train(self.global_features)
            index.add(self.global_features)

        # Query the same features as the dataset itself
        distances, indices = index.search(self.global_features, n_neighbors)

        # Majority voting for k-NN prediction
        predicted_labels = []
        for idx_list in indices:
            neighbor_labels = self.global_labels[idx_list]
            predicted_label = np.bincount(neighbor_labels).argmax()
            predicted_labels.append(predicted_label)

        predicted_labels = np.array(predicted_labels)

        # Keep correctly classified samples
        self.correct_indices = np.where(predicted_labels == self.global_labels)[0].tolist() 
        print(f"Cleaned global datastore: {len(self.correct_indices)} samples retained ({original_global_size - len(self.correct_indices)} samples removed).")

        
    def send_relevant_features(self, client_id, client_labels):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)

        # Exclude the current client's features
        client_indices = set(range(*self.all_clients_indices.get(client_id, (0, 0))))
        
        # Sanity check: Ensure correct indices mapped to client's data
        own_features = self.global_features[list(client_indices)]
        own_labels = self.global_labels[list(client_indices)]
        print(f"Number of Client {client_id}'s own features: {len(own_features)}")
        assert len(own_features) == len(own_labels), "Mismatch in own features and labels length"

        # Send back features relevant to the client's local labels
        relevant_indices = [i for i, label in enumerate(self.global_labels) 
                            if i not in client_indices and label in client_labels] 
        
        if len(relevant_indices) > self.max_global_features_per_client:
            relevant_indices = np.random.choice(relevant_indices, size=self.max_global_features_per_client, replace=False)
        
        selected_features = self.global_features[relevant_indices]
        selected_labels = self.global_labels[relevant_indices]
        # Final sanity checks
        assert not any(i in client_indices for i in relevant_indices), (
            f"Client {client_id} received its own data back!"
        )
        assert len(selected_features) == len(selected_labels), (
            "Mismatch between selected features and labels"
        )

        return selected_features, selected_labels
    
    
    def send_relevant_features_class_balanced(self, client_id, client_labels):
        """
        Sends class-balanced global features based on the client's local class distribution.
        The distribution is prioritized towards underrepresented classes.
        
        :param client_labels: array-like, labels present in the client's local data.
        :return: Tuple (selected_features, selected_labels) to augment client's local datastore.
        """
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)

        # Compute class distribution in client's local data
        class_counts = Counter(client_labels)
        total_count = sum(class_counts.values())
        
        # Calculate sampling weights (higher for underrepresented classes)
        class_weights = {label: (total_count - count) / total_count for label, count in class_counts.items()}

        # Identify missing classes (not in client_labels)
        global_classes = set(self.global_labels)  # Unique global classes
        missing_classes = global_classes - set(client_labels)

        # Map class indices
        class_indices = {label: np.where(self.global_labels == label)[0] for label in global_classes}
        client_indices = set(range(*self.all_clients_indices.get(client_id, (0, 0))))

        # Exclude client's own features from the global selection
        relevant_indices = []

        # Handle underrepresented existing classes
        for label in class_weights:
            if label in class_indices:
                filtered_indices = [i for i in class_indices[label] if i not in client_indices and i in self.correct_indices]
                num_samples = int(class_weights[label] * self.max_global_features_per_client)
                num_samples = min(num_samples, len(filtered_indices))
                if num_samples > 0:
                    sampled_indices = np.random.choice(filtered_indices, size=num_samples, replace=False)
                    relevant_indices.extend(sampled_indices)

        # Handle missing classes (give them equal weight to balance the client's distribution)
        if missing_classes:
            missing_class_indices = [i for label in missing_classes for i in class_indices.get(label, []) if i in self.correct_indices]
            num_samples_missing = min(self.max_global_features_per_client // 2, len(missing_class_indices))
            if num_samples_missing > 0:
                sampled_missing_indices = np.random.choice(missing_class_indices, size=num_samples_missing, replace=False)
                relevant_indices.extend(sampled_missing_indices)

        
        if len(relevant_indices) > self.max_global_features_per_client:
            relevant_indices = np.random.choice(relevant_indices, size=self.max_global_features_per_client, replace=False)

        selected_features = self.global_features[relevant_indices]
        selected_labels = self.global_labels[relevant_indices]
        # Final sanity checks
        assert not any(i in client_indices for i in relevant_indices), (
            f"Client {client_id} received its own data back!"
        )
        assert len(selected_features) == len(selected_labels), (
            "Mismatch between selected features and labels"
        )
        return selected_features, selected_labels


class NoCommunicationAggregator(Aggregator):
    r"""Clients do not communicate. Each client work locally
    Adapted from: https://github.com/omarfoq/knn-per/tree/main
    """
    def mix(self):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step()

        self.c_round += 1

    def toggle_client(self, client_id, mode):
        pass

