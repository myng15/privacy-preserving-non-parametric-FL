"""Adapted from: https://github.com/omarfoq/knn-per/tree/main"""

import os
import torch
import warnings
import argparse
from abc import ABC, abstractmethod


class ArgumentsManager(ABC):
    r"""This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.

    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(description=__doc__)
        self.args = None
        self.initialized = False

        self.parser.add_argument(
            'experiment',
            help='name of experiment, possible are: e.g.  "organsmnist", "dermamnist" ',
            type=str
        )
        self.parser.add_argument(
            '--backbone',
            help='the backbone used to extract feature embeddings;'
                 'default=vit_base_patch14_dinov2.lvd142m',
            type=str,
            default="vit_base_patch14_dinov2.lvd142m"
        )
        self.parser.add_argument(
            '--model_name',
            help='the name of the model to be used',
            type=str,
            default="linear"
        )
        self.parser.add_argument(
            '--aggregator_type',
            help='aggregator type; possible are "centralized", "centralized_linear", "fedknn"', 
            type=str,
            default="centralized"
        )
        self.parser.add_argument(
            '--client_type',
            help='client type; possible are "normal", "fedknn"; default is "normal"',
            type=str,
            default="normal"
        )
        self.parser.add_argument(
            '--s_frac',
            help='fraction of the dataset to be used; default: 1.0;',
            type=float,
            default=1.0
        )
        self.parser.add_argument(
            '--val_frac',
            help='fraction of each client\'s train dataset used for validation; default=0.1',
            type=float,
            default=0.1
        )
        self.parser.add_argument(
            '--bz',
            help='batch_size; default is 1',
            type=int,
            default=1
        )
        self.parser.add_argument(
            '--device',
            help='device to use, either cpu or cuda; default is cpu',
            type=str,
            default="cpu"
        )
        self.parser.add_argument(
            "--input_dimension",
            help='input dimension ofr two layers linear model',
            type=int,
            default=150
        )
        self.parser.add_argument(
            "--hidden_dimension",
            help='hidden dimension for two layers linear model',
            type=int,
            default=10
        )
        self.parser.add_argument(
            "--seed",
            help='random seed; if not specified the system clock is used to generate the seed',
            type=int,
            default=argparse.SUPPRESS
        )
        self.parser.add_argument(
            "--results_dir",
            help='directory to save the results; if not passed, it is set using arguments',
            default=argparse.SUPPRESS
        )

    def parse_arguments(self, args_list=None):
        if args_list:
            args = self.parser.parse_args(args_list)
        else:
            args = self.parser.parse_args()

        self.args = args

        if self.args.device == "cuda" and not torch.cuda.is_available():
            self.args.device = "cpu"
            warnings.warn("CUDA is not available, device is automatically set to \"CPU\"!", RuntimeWarning)

        self.initialized = True

    @abstractmethod
    def args_to_string(self):
        pass


class TrainArgumentsManager(ArgumentsManager):
    def __init__(self):
        super(TrainArgumentsManager, self).__init__()

        self.parser.add_argument(
            '--algorithm',
            help='algorithm for optimizing FedAvg; possible are "normal", "mixup", "fedadam", "fedadagrad", "fedprox"; default is "normal"',
            type=str,
            default="normal"
        )
        self.parser.add_argument(
            '--sampling_rate',
            help='proportion of clients to be used at each round; default is 1.0',
            type=float,
            default=1.0
        )
        self.parser.add_argument(
            '--n_rounds',
            help='number of communication rounds; default is 1',
            type=int,
            default=1
        )
        self.parser.add_argument(
            '--local_steps',
            help='number of local steps before communication; default is 1',
            type=int,
            default=5 
        )
        self.parser.add_argument(
            '--log_freq',
            help='frequency of writing logs; defaults is 1',
            type=int,
            default=1
        )
        self.parser.add_argument(
            '--eval_freq',
            help='frequency of logging client evaluation results; defaults is 99999 (i.e. only once before and once after training)',
            type=int,
            default=99999
        )
        self.parser.add_argument(
            '--optimizer',
            help='optimizer to be used for the training; default is sgd',
            type=str,
            default="sgd"
        )
        self.parser.add_argument(
            "--lr",
            type=float,
            help='learning rate; default is 1e-3',
            default=1e-3
        )
        self.parser.add_argument(
            "--lr_scheduler",
            help='learning rate decay scheme to be used;'
                 ' possible are "sqrt", "linear", "cosine_annealing" and "constant"(no learning rate decay);'
                 'default is "constant"',
            type=str,
            default="constant"
        )
        self.parser.add_argument(
            "--mu",
            help='proximal / penalty term weight, used when --optimizer=`prox_sgd` also used with L2SGD; '
                 'default is `0.`',
            type=float,
            default=0
        )
        self.parser.add_argument(
            '--validation',
            help='if chosen the validation part will be used instead of test part;'
                 ' make sure to use `val_frac > 0` in `generate_data.py`;',
            action='store_true'
        )
        self.parser.add_argument(
            "--logs_dir",
            help='directory to write logs; if not passed, it is set using arguments',
            default=argparse.SUPPRESS
        )
        self.parser.add_argument(
            "--chkpts_dir",
            help='directory to save checkpoints once the training is over; if not specified checkpoints are not saved',
            default=argparse.SUPPRESS
        )
        self.parser.add_argument(
            "--verbose",
            help='verbosity level, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`;',
            type=int,
            default=0
        )

    def args_to_string(self):
        """
        Transform experiment's arguments into a string

        :return: string

        """
        args_string = ""

        args_to_show = ["experiment"]
        for arg in args_to_show:
            args_string = os.path.join(args_string, str(getattr(self.args, arg)))

        return args_string


class TestArgumentsManager(ArgumentsManager):
    def __init__(self):
        super(TestArgumentsManager, self).__init__()

        self.parser.add_argument(
            'strategy',
            help='name of the strategy used to build the datastore;'
                 ' possible are; `random`',
            type=str
        )
        self.parser.add_argument(
            '--chkpts_path',
            help='path to checkpoints file;',
            type=str,
            default=""
        )
        self.parser.add_argument(
            '--fedavg_chkpts_dir',
            help='path to checkpoints file;',
            type=str,
            default=""
        )
        self.parser.add_argument(
            '--classifier', 
            help='the client\'s own classifier', 
            type=str, 
            default='knn'
        )
        self.parser.add_argument(
            '--anonymizer', 
            help='the data anonymizer', 
            type=str, 
            default=None
        )
        self.parser.add_argument(
            "--use_pretrained_cvae_fedavg",
            help="use pretrained CVAE_FEDAVG checkpoints",
            action="store_true",
            default=False,
        )
        self.parser.add_argument(
            '--n_fedavg_rounds',
            help='number of communication rounds for FedAvg training of generative models (anonymizers); default is 1',
            type=int,
            default=1
        )
        self.parser.add_argument(
            "--anonymizer_lr",
            type=float,
            help='learning rate for anonymizer (both local and FedAvg training); default is 1e-3',
            default=1e-3
        )
        self.parser.add_argument(
            "--cvae_beta",
            type=float,
            help='beta for CVAE training; default is 0.1',
            default=0.1
        )
        self.parser.add_argument(
            "--cvae_var",
            type=float,
            help='variance for CVAE generation; default is 1.0',
            default=1.0
        )
        self.parser.add_argument(
            '--total_generated_factor', 
            help='the factor by which the total number of generated samples is multiplied in relation to the original sample size',
            type=float, 
            default=None
        )
        self.parser.add_argument(
            "--augmentation_strategy",
            type=str,
            help='the augmentation strategy using the generated data;'
                 'possible are "replicate", "uniform", "uniform_all_classes" and "inverse"' 
                 'default=uniform',
            default="uniform"
        )
        self.parser.add_argument(
            '--global_sampling',
            help='the method of sampling global features to be sent to each client;'
                 'possible are "random" and "class_balanced"' 
                 'default=random',
            type=str,
            default="random"
        )
        self.parser.add_argument(
            '--clustering',
            help='the method of computing cluster centroids of local features to be sent to global aggregator;'
                 'possible are None, "unsupervised", "class_based" and "class_based_torch"'
                 'default=None', 
            type=str,
            default=None
        )
        self.parser.add_argument(
            "--n_clusters",
            help="number of k_means clusters;",
            type=int,
            default=None
        )
        self.parser.add_argument(
            '--centroids_dp_noise',
            help='the type of DP noise applied to cluster centroids of local features to be sent to global aggregator;'
                 'possible are "laplacian", "gaussian" or None'
                 'default=None', 
            type=str,
            default=None
        )
        self.parser.add_argument(
            "--return_centroids",
            help="whether to return_centroids only or the entire (k-Same style) anonymized dataset",
            action="store_true",
            default=False,
        )
        self.parser.add_argument(
            '--k_same', 
            help='the k value used for k-Same',
            type=int, 
            default=None
        )
        self.parser.add_argument(
            '--capacities_grid_resolution',
            help='the resolution of the capacities, the smaller it is the higher the resolution;'
                 ' should be smaller then 1.; higher value of resolution requires more computation time.',
            type=float
        )
        self.parser.add_argument(
            '--weights_grid_resolution',
            help='the resolution of the weights grid, the smaller it is the higher the resolution;'
                 ' should be smaller then 1.; higher value of resolution requires more computation time.',
            type=float
        )
        self.parser.add_argument(
            "--n_neighbors",
            help="number of neighbours used in nearest neighbours retrieval;",
            type=int,
            default=3
        )
        self.parser.add_argument(
            '--knn_metric',
            help='the distance metric for knn search;'
                 'possible are "euclidean" and "cosine"' 
                 'default=euclidean',
            type=str,
            default="euclidean"
        )
        self.parser.add_argument(
            '--knn_weights',
            help='the method of weighing knn outputs;'
                 'possible are "inverse_distances" and "gaussian_kernel"' 
                 'default=gaussian_kernel',
            type=str,
            default="gaussian_kernel"
        )
        self.parser.add_argument(
            '--gaussian_kernel_scale',
            help='the length scale of kernel; default=1.0',
            type=float,
            default=1.0
        )
        self.parser.add_argument(
            '--classifier_optimizer', 
            help='optimizer for the client\'s own classifier', choices=['sgd', 'adam'],
            type=str, 
            default='sgd'
        )
        self.parser.add_argument(
            "--local_epochs",
            help="number of local training epochs for linear/MLP classifiers",
            type=int,
            default=100
        )
        self.parser.add_argument(
            "--verbose",
            help='verbosity level, `0` to quiet, `1` to show samples statistics;',
            type=int,
            default=0
        )
        self.parser.add_argument(
            "--enable_dp",
            help="enable differential privacy training",
            action="store_true",
            default=False,
        )
        self.parser.add_argument(
            "--max_feat_norm",
            help="clip features to this norm (default 1.0) in DP noise adding schemes",
            type=float,
            default=1.0,
        )
        self.parser.add_argument(
            "--max_grad_norm",
            help="clip per-sample gradients to this norm (default 1.0) in DP training",
            type=float,
            default=1.0,
        )
        self.parser.add_argument(
            "--noise_multiplier",
            help="Noise multiplier (default 1.0) in DP training",
            type=float,
            default=1.0,
        )
        self.parser.add_argument(
            "--epsilon",
            help="Epsilon (default 50.0) in DP training",
            type=float,
            default=50.0,
        )
        self.parser.add_argument(
            "--delta",
            help="target delta (default: 1e-5) in DP training",
            type=float,
            default=1e-5,
        )

    def args_to_string(self):
        """
        Transform experiment's arguments into a string

        :return: string

        """
        args_string = ""

        args_to_show = ["experiment", "strategy", "n_neighbors"]
        for arg in args_to_show:
            args_string = os.path.join(args_string, str(getattr(self.args, arg)))

        return args_string


class PlotsArgumentsManager(ArgumentsManager):
    def __init__(self):
        super(PlotsArgumentsManager, self).__init__()

        self.parser = argparse.ArgumentParser(description=__doc__)
        self.args = None
        self.initialized = False

        self.parser.add_argument(
            'plot_name',
            help='name of the plot, possible are:'
                 '{"capacity_effect", "weight_effect", "hetero_effect", "n_neighbors_effect"}'
        )
        self.parser.add_argument(
            '--results_dir',
            help='directory to the results; should contain files `all_scores.npy`, `capacities_grid.npy`,'
                 '`weights_grid.npy` abd `capacities_grid.npy`;',
            type=str
        )

    def args_to_string(self):
        pass

    def parse_arguments(self, args_list=None):
        if args_list:
            args = self.parser.parse_args(args_list)
        else:
            args = self.parser.parse_args()

        self.args = args

        self.initialized = True
