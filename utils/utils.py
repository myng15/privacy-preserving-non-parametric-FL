from aggregator import *
from client import *
from learners.learner import *
from models.classification_models import *
from datasets import *

from .constants import *
from .metrics import *
from .optim import *

from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from tqdm import tqdm

import os

def seed_everything(seed=42):
    """
    Ensure reproducibility.
    :param seed: Integer defining the seed number.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_embeddings(experiment_):
    raw_data_path = "data/" + experiment_ + "/database/"

    # Load embeddings and labels from the .npz files
    train_data = np.load(os.path.join(raw_data_path, 'train.npz'))
    test_data = np.load(os.path.join(raw_data_path, 'test.npz'))

    train_embeddings = train_data['embeddings']
    test_embeddings = test_data['embeddings']

    type_ = LOADER_TYPE[experiment_]

    if type_ == "medmnist":
        train_labels = train_data['labels'].squeeze().astype(int)
        test_labels = test_data['labels'].squeeze().astype(int)
    else:
        train_labels = train_data['labels']
        test_labels = test_data['labels']

    if type_ == "camelyon17":
        train_metadata = train_data['metadata']
        test_metadata = test_data['metadata']
    else:
        train_metadata = None
        test_metadata = None

    embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
    labels = np.concatenate([train_labels, test_labels], axis=0)

    # Combine train and test embeddings and labels into a single dataset
    if type_ == "camelyon17":
        metadatas = np.concatenate([train_metadata, test_metadata], axis=0)
    else:
        metadatas = None

    if type_ == "camelyon17":
        return embeddings, labels, metadatas
    else:
        return embeddings, labels


def get_data_dir(experiment_name):
    """
    returns a string representing the path where to find the datafile corresponding to the experiment
    :param experiment_name: name of the experiment
    :return: str
    """
    data_dir = os.path.join("data", experiment_name, "all_clients_data")

    return data_dir


def get_loader(type_, aggregator_, path, batch_size, train, inputs=None, targets=None):
    """
    constructs a torch.utils.DataLoader object from the given path
    :param type_: type of the dataset,
     `femnist` and `shakespeare`
    :param path: path to the data file
    :param batch_size:
    :param train: flag indicating if train loader or test loader
    :param inputs: tensor storing the input data; default is None
    :param targets: tensor storing the labels; default is None
    :return: torch.utils.DataLoader
    """
    if type_ == "camelyon17": 
        dataset = SubCAMELYON17(path, aggregator_, camelyon17_data=inputs, camelyon17_targets=targets) 

    else:
        raise NotImplementedError(f"{type_} not recognized type; possible are {list(LOADER_TYPE.keys())}")

    if len(dataset) == 0:
        return

    # drop last batch
    drop_last = (len(dataset) > batch_size) and train

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last, num_workers=NUM_WORKERS) 


def get_loaders(experiment_, aggregator_, data_dir, batch_size, is_validation):
    """
    Adapted from: https://github.com/omarfoq/knn-per/tree/main

    constructs lists of `torch.utils.DataLoader` object from the given files in `root_path`;
     corresponding to `train_iterator`, `val_iterator` and `test_iterator`;
     `val_iterator` iterates on the same dataset as `train_iterator`, the difference is only in drop_last
    :param type_: type of the dataset;
    :param data_dir: directory of the data folder
    :param batch_size:
    :param is_validation: (bool) if `True` validation part is used as test
    :return:
        train_iterator, val_iterator, test_iterator
        (List[torch.utils.DataLoader], List[torch.utils.DataLoader], List[torch.utils.DataLoader])
    """
    type_ = LOADER_TYPE[experiment_]

    if type_ == "mnist": 
        if aggregator_ == "centralized":
            inputs, targets = get_mnist() 
        else:
            inputs, targets = load_embeddings(experiment_)
    elif type_ == "medmnist": 
        if aggregator_ == "centralized":
            inputs, targets = None # Dummy value, as we are not dealing with this case (using raw data)
        else:
            inputs, targets = load_embeddings(experiment_)

    else:
        inputs, targets = None, None

    train_iterators, val_iterators, test_iterators = [], [], []

    for client_id, client_dir in enumerate(tqdm(os.listdir(data_dir))):
        client_data_path = os.path.join(data_dir, client_dir)

        train_iterator = \
            get_loader(
                type_=type_,
                aggregator_=aggregator_, 
                path=os.path.join(client_data_path, "train.npz") if type_ in ["camelyon17"] 
                     else os.path.join(client_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=True
            )

        val_iterator = \
            get_loader(
                type_=type_,
                aggregator_=aggregator_, 
                path=os.path.join(client_data_path, "val.npz") if type_ in ["camelyon17"] 
                     else os.path.join(client_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            )

        test_set = "val" if is_validation else "test"

        test_iterator = \
            get_loader(
                type_=type_,
                aggregator_=aggregator_, 
                path=os.path.join(client_data_path, "test.npz") if type_ in ["camelyon17"]
                     else os.path.join(client_data_path, f"{test_set}{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            )

        train_iterators.append(train_iterator)
        val_iterators.append(val_iterator)
        test_iterators.append(test_iterator)

    return train_iterators, val_iterators, test_iterators


def get_model(name, model_name, device, input_dimension=None, hidden_dimension=None, chkpts_path=None):
    """
    Adapted from: https://github.com/omarfoq/knn-per/tree/main

    create model and initialize it from checkpoints

    :param name: experiment's name

    :param model_name: the name of the model to be used (if training on raw data, not embedding-based data),
            possible are mobilenet and resnet

    :param device: either cpu or cuda

    :param input_dimension:

    :param hidden_dimension:

    :param chkpts_path: path to chkpts; if specified the weights of the model are initialized from chkpts,
                        otherwise the weights are initialized randomly; default is None.
    """
    if name == "mnist":
        if model_name == "mobilenet":
            model = get_mobilenet(n_classes=10, pretrained=True)
        elif model_name == "linear":
            model = LinearLayer(
                input_dimension=input_dimension,
                num_classes=10
            )
        elif model_name == "mlp":
            model = MultiLayerPerceptron(
                input_dimension=input_dimension,
                num_classes=10
            )
        else:
            error_message = f"{model_name } is not a possible arrival process, available are:"
            for model_name_ in ALL_MODELS:
                error_message += f" `{model_name_};`"

            raise NotImplementedError(error_message)
    elif name in ["organamnist", "organcmnist", "organsmnist"]:
        if model_name == "mobilenet":
            model = get_mobilenet(n_classes=11, pretrained=True)
        elif model_name == "linear":
            model = LinearLayer(
                input_dimension=input_dimension,
                num_classes=11
            )
        elif model_name == "mlp":
            model = MultiLayerPerceptron(
                input_dimension=input_dimension,
                num_classes=11
            )
        else:
            error_message = f"{model_name} is not a possible model, available are:"
            for model_name_ in ALL_MODELS:
                error_message += f" `{model_name_};`"

            raise NotImplementedError(error_message)
    elif name == "dermamnist":
        if model_name == "mobilenet":
            model = get_mobilenet(n_classes=7, pretrained=True)
        elif model_name == "linear":
            model = LinearLayer(
                input_dimension=input_dimension,
                num_classes=7
            )
        elif model_name == "mlp":
            model = MultiLayerPerceptron(
                input_dimension=input_dimension,
                num_classes=7
            )
        else:
            error_message = f"{model_name} is not a possible model, available are:"
            for model_name_ in ALL_MODELS:
                error_message += f" `{model_name_};`"

            raise NotImplementedError(error_message)
    elif name == "retinamnist":
        if model_name == "mobilenet":
            model = get_mobilenet(n_classes=5, pretrained=True)
        elif model_name == "linear":
            model = LinearLayer(
                input_dimension=input_dimension,
                num_classes=5
            )
        elif model_name == "mlp":
            model = MultiLayerPerceptron(
                input_dimension=input_dimension,
                num_classes=5
            )
        else:
            error_message = f"{model_name} is not a possible model, available are:"
            for model_name_ in ALL_MODELS:
                error_message += f" `{model_name_};`"

            raise NotImplementedError(error_message)
    elif name == "pathmnist":
        if model_name == "mobilenet":
            model = get_mobilenet(n_classes=9, pretrained=True)
        elif model_name == "linear":
            model = LinearLayer(
                input_dimension=input_dimension,
                num_classes=9
            )
        elif model_name == "mlp":
            model = MultiLayerPerceptron(
                input_dimension=input_dimension,
                num_classes=9
            )
        else:
            error_message = f"{model_name} is not a possible model, available are:"
            for model_name_ in ALL_MODELS:
                error_message += f" `{model_name_};`"

            raise NotImplementedError(error_message)
    elif name == "bloodmnist":
        if model_name == "mobilenet":
            model = get_mobilenet(n_classes=9, pretrained=True)
        elif model_name == "linear":
            model = LinearLayer(
                input_dimension=input_dimension,
                num_classes=8
            )
        elif model_name == "mlp":
            model = MultiLayerPerceptron(
                input_dimension=input_dimension,
                num_classes=8
            )
        else:
            error_message = f"{model_name} is not a possible model, available are:"
            for model_name_ in ALL_MODELS:
                error_message += f" `{model_name_};`"

            raise NotImplementedError(error_message)
    elif name == "camelyon17" or name == "pneumoniamnist":
        if model_name == "mobilenet":
            model = get_mobilenet(n_classes=2, pretrained=True)
        elif model_name == "linear":
            model = LinearLayer(
                input_dimension=input_dimension,
                num_classes=2
            )
        elif model_name == "mlp":
            model = MultiLayerPerceptron(
                input_dimension=input_dimension,
                num_classes=2
            )
        else:
            error_message = f"{model_name} is not a possible model, available are:"
            for model_name_ in ALL_MODELS:
                error_message += f" `{model_name_};`"

            raise NotImplementedError(error_message)
    
    else:
        raise NotImplementedError(
            f"{name} is not available!"
        )

    if chkpts_path is not None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model.load_state_dict(torch.load(chkpts_path, map_location=map_location, weights_only=True)['model_state_dict'])
        except KeyError:
            try:
                model.load_state_dict(torch.load(chkpts_path, map_location=map_location, weights_only=True)['net'])
            except KeyError:
                model.load_state_dict(torch.load(chkpts_path, map_location=map_location, weights_only=True))

    model = model.to(device)
    
    return model


def get_client(
        client_type,
        learner,
        train_iterator,
        val_iterator,
        test_iterator,
        logger,
        local_steps=None,
        client_id=None,
        save_path=None,
        args=None,
        features_dimension=None,
        capacity=None,
        rng=None,
):
    """
    Adapted from: https://github.com/omarfoq/knn-per/tree/main

    :param client_type:
    :param learner:
    :param train_iterator:
    :param val_iterator:
    :param test_iterator:
    :param logger:
    :param local_steps:
    :param client_id:
    :param save_path:
    :param k: number of neighbours used in KNNClient; default is None
    :param interpolate_logits: if selected logits are interpolated instead of probabilities
    :param features_dimension: feature space dimension of the embedding space;
                                only used with KNNClient; default is None
    :param num_classes: number of classes; only used with KNNClient; default is None
    :param capacity: datastore capacity; only used with KNNClient; default is None
    :param strategy: strategy to build the datastore; only used with KNNClient; default is None
    :param rng: random number generator; only used with KNNClient; default is None

    :return:
        Client

    """
    if client_type == "fedknn":
        return KNNClient(
            learner=None,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            id_=client_id,
            args_=args,
            k=args.n_neighbors,
            n_clusters=args.n_clusters,
            features_dimension=features_dimension,
            num_classes=N_CLASSES[args.experiment],
            capacity=-1,
            strategy=args.strategy,
            rng=rng,
            knn_weights=args.knn_weights,
            gaussian_kernel_scale=args.gaussian_kernel_scale,
            device=args.device,
            seed=args.seed
        )
    else:
        return Client(
            learner=learner,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            id_=client_id,
            save_path=save_path
        )


def get_learner(
        name,
        model_name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        mu,
        n_rounds,
        seed,
        algorithm=None,
        input_dimension=None,
        hidden_dimension=None,
        chkpts_path=None,
):
    """
    Adapted from: https://github.com/omarfoq/knn-per/tree/main

    constructs the learner corresponding to an experiment for a given seed

    :param name: name of the experiment to be used; 

    :param model_name: the name of the model to be used (if training on raw data, not embedding-based data),
            possible are mobilenet and resnet

    :param device: used device; possible `cpu` and `cuda`

    :param optimizer_name: passed as argument to utils.optim.get_optimizer

    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler

    :param initial_lr: initial value of the learning rate

    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`

    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;

    :param seed:

    :param input_dimension:

    :param hidden_dimension:

    :param chkpts_path: path to chkpts; if specified the weights of the model are initialized from chkpts,
            otherwise the weights are initialized randomly; default is None.

    :return: Learner

    """
    torch.manual_seed(seed)

    criterion = nn.CrossEntropyLoss(reduction="none").to(device)
    metric = accuracy
    
    if name in ["camelyon17"]:
        is_binary_classification = True
    else:
        is_binary_classification = False

    model = \
        get_model(
            name=name,
            model_name=model_name,
            device=device,
            chkpts_path=chkpts_path,
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension
        )

    optimizer =\
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr_initial=initial_lr,
            mu=mu
        )

    lr_scheduler =\
        get_lr_scheduler(
            optimizer=optimizer,
            scheduler_name=scheduler_name,
            n_rounds=n_rounds
        )

    # SWAD
    client_algorithm = ClientAlgorithm(model, device, is_swad=(algorithm == "swad"))

    return Learner(
        model=model,
        model_name=model_name,
        criterion=criterion,
        metric=metric,
        device=device,
        optimizer=optimizer,
        algorithm=algorithm, 
        client_algorithm=client_algorithm, 
        lr_scheduler=lr_scheduler,
        is_binary_classification=is_binary_classification
    )


def get_aggregator(
        aggregator_type,
        clients,
        algorithm=None,
        lr=None, # for FedAdam etc.
        global_learner=None,
        sampling_rate=None,
        log_freq=None,
        global_train_logger=None,
        global_test_logger=None,
        test_clients=None,
        verbose=None,
        features_dimension=None,
        seed=None
):
    """
    Adapted from: https://github.com/omarfoq/knn-per/tree/main

    :param aggregator_type:
    :param clients:
    :param global_learner:
    :param sampling_rate:
    :param log_freq:
    :param global_train_logger:
    :param global_test_logger:
    :param test_clients
    :param verbose: level of verbosity
    :param seed: default is None
    :return:

    """
    if aggregator_type == "local":
        return NoCommunicationAggregator(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "centralized" or aggregator_type == "centralized_linear": 
        return CentralizedAggregator(
            clients=clients,
            algorithm=algorithm,
            lr=lr,
            global_learner=global_learner,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "fedknn": 
        return FedKNNAggregator(
            num_clients=len(clients), 
            features_dimension=features_dimension,
            seed=seed
        )
    else:
        raise NotImplementedError(
            f"{aggregator_type} is not available!"
            f" Possible are: `local` and `centralized`."
        )


