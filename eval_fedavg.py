"""Adapted from: https://github.com/omarfoq/knn-per/tree/main"""

from collections import defaultdict
from utils.utils import *
from utils.constants import *
from utils.args import TrainArgumentsManager

from torch.utils.tensorboard import SummaryWriter


def init_clients(args_, data_dir, logs_dir, chkpts_dir):
    """
    initialize clients from data folders

    :param args_:
    :param data_dir: path to directory containing data folders
    :param logs_dir: directory to save the logs
    :param chkpts_dir: directory to save chkpts
    :return: List[Client]

    """
    os.makedirs(chkpts_dir, exist_ok=True)

    print("===> Initializing clients..")
    
    # FOR DATASETS WITH NATURAL DOMAINS
    if args_.experiment in ["camelyon17", "fitzpatrick17k", "epistroma", "covidfl"]:
        train_loaders, val_loaders, test_loaders = get_loaders(
            experiment_=args_.experiment,
            aggregator_=args_.aggregator_type, 
            data_dir=data_dir,
            batch_size=args_.bz,
            is_validation=args_.validation,
        )

        num_clients = len(train_loaders)
        clients_ = []

        # Initialize a client object for every `train loader` and `test loader`
        for client_id, (train_loader, val_loader, test_loader) in enumerate(tqdm(
            zip(train_loaders, val_loaders, test_loaders), 
            total=num_clients
        )):
            if train_loader is None or test_loader is None:
                continue

            if args_.verbose > 0:
                print(f"[Client ID: {client_id}] N_Train: {len(train_loader.dataset)} | N_Val: {len(val_loader.dataset)} | N_Test: {len(test_loader.dataset)}")

            learner =\
                get_learner(
                    name=args_.experiment,
                    model_name=args_.model_name,
                    device=args_.device,
                    optimizer_name=args_.optimizer,
                    scheduler_name=args_.lr_scheduler,
                    initial_lr=args_.lr,
                    n_rounds=args_.n_rounds,
                    seed=args_.seed,
                    algorithm=args_.algorithm,
                    input_dimension=EMBEDDING_DIM[args_.backbone], 
                    hidden_dimension=None, 
                    mu=args_.mu 
                )

            logs_path = os.path.join(logs_dir, "client_{}".format(client_id))
            os.makedirs(logs_path, exist_ok=True)
            logger = SummaryWriter(logs_path)

            client = get_client(
                client_type=args_.client_type,
                learner=learner,
                train_iterator=train_loader,
                val_iterator=val_loader,
                test_iterator=test_loader,
                logger=logger,
                local_steps=args_.local_steps,
                client_id=client_id,
                save_path=os.path.join(chkpts_dir, "client_{}.pt".format(client_id))
            )

            clients_.append(client)

    
    # FOR DATASETS WITH ARTIFICIAL CLIENT SPLITS
    else: 
        train_loaders, _, test_loaders = get_loaders(
            experiment_=args_.experiment,
            aggregator_=args_.aggregator_type, 
            data_dir=data_dir,
            batch_size=args_.bz,
            is_validation=args_.validation,
        )

        num_clients = len(train_loaders)
        clients_ = []

        # Initialize a client object for every `train loader` and `test loader`
        for client_id, (train_loader, test_loader) in enumerate(tqdm(
            zip(train_loaders, test_loaders), 
            total=num_clients
        )):
            if train_loader is None or test_loader is None:
                continue

            # WITH STRATIFIED SPLIT
            dataset = train_loader.dataset

            # Group indices by label
            label_to_indices = defaultdict(list)
            for idx in range(len(dataset)):
                _, label, _ = dataset[idx]
                label_to_indices[label.item()].append(idx)

            train_indices = []
            val_indices = []

            # Ensure deterministic and stratified split per label
            for label, indices in label_to_indices.items():
                np.random.seed(args_.seed + label)  # Unique seed per label for deterministic split
                np.random.shuffle(indices)

                if len(indices) == 1:
                    train_indices.extend(indices)  # Only one sample with this label, put it in training
                else:
                    split_point = int((1 - args_.val_frac) * len(indices))
                    train_indices.extend(indices[:split_point])
                    val_indices.extend(indices[split_point:])
            

            # Create train and validation datasets
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)

            # Create data loaders
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_loader.batch_size)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_loader.batch_size)


            if args_.verbose > 0:
                print(f"[Client ID: {client_id}] N_Train: {len(train_loader.dataset)} | N_Val: {len(val_loader.dataset)} | N_Test: {len(test_loader.dataset)}")

            learner =\
                get_learner(
                    name=args_.experiment,
                    model_name=args_.model_name,
                    device=args_.device,
                    optimizer_name=args_.optimizer,
                    scheduler_name=args_.lr_scheduler,
                    initial_lr=args_.lr,
                    n_rounds=args_.n_rounds,
                    seed=args_.seed,
                    algorithm=args_.algorithm,
                    input_dimension=EMBEDDING_DIM[args_.backbone], 
                    hidden_dimension=None, 
                    mu=args_.mu 
                )

            logs_path = os.path.join(logs_dir, "client_{}".format(client_id))
            os.makedirs(logs_path, exist_ok=True)
            logger = SummaryWriter(logs_path)

            client = get_client(
                client_type=args_.client_type,
                learner=learner,
                train_iterator=train_loader,
                val_iterator=val_loader,
                test_iterator=test_loader,
                logger=logger,
                local_steps=args_.local_steps,
                client_id=client_id,
                save_path=os.path.join(chkpts_dir, "client_{}.pt".format(client_id))
            )

            clients_.append(client)

    return clients_


def run(arguments_manager_):
    """

    :param arguments_manager_:
    :type arguments_manager_: ArgumentsManager

    """
    
    if not arguments_manager_.initialized:
        arguments_manager_.parse_arguments()

    args_ = arguments_manager_.args

    seed_everything(args_.seed)
    
    data_dir = get_data_dir(args_.experiment)

    if "logs_dir" in args_:
        logs_dir = args_.logs_dir
    else:
        logs_dir = os.path.join("logs", arguments_manager_.args_to_string()) 

    if "chkpts_dir" in args_:
        chkpts_dir = args_.chkpts_dir
    else:
        chkpts_dir = os.path.join("chkpts", arguments_manager_.args_to_string()) 

    print("==> Clients initialization starts...")
    clients = \
        init_clients(
            args_,
            data_dir=os.path.join(data_dir, "train"),
            logs_dir=os.path.join(logs_dir, "train"),
            chkpts_dir=os.path.join(chkpts_dir, "train")
        )     

    print("==> Test Clients initialization starts...")
    test_clients = \
        init_clients(
            args_,
            data_dir=os.path.join(data_dir, "test"),
            logs_dir=os.path.join(logs_dir, "test"),
            chkpts_dir=os.path.join(chkpts_dir, "test")
        )

    logs_path = os.path.join(logs_dir, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_dir, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    global_learner = \
        get_learner(
            name=args_.experiment,
            model_name=args_.model_name, 
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu,
            input_dimension=EMBEDDING_DIM[args_.backbone], 
            hidden_dimension=None 
        )

    aggregator = \
        get_aggregator(
            aggregator_type=args_.aggregator_type, 
            clients=clients,
            algorithm=args_.algorithm,
            lr=args_.lr,
            global_learner=global_learner,
            sampling_rate=args_.sampling_rate,
            log_freq=args_.log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            verbose=args_.verbose,
            seed=args_.seed
        )
    

    all_client_results_ = []
    all_test_client_results_ = []
    all_eval_rounds_ = []

    aggregator.write_logs()

    print("Training..")
    for ii in tqdm(range(args_.n_rounds)):
        aggregator.mix()

        if (ii % args_.log_freq) == (args_.log_freq - 1):
            aggregator.save_state(chkpts_dir)
            aggregator.write_logs(epoch=ii, max_epochs=args_.n_rounds, save_path=args_.results_dir)

    aggregator.save_state(chkpts_dir)

    all_client_results_ = np.array(all_client_results_)
    all_test_client_results_ = np.array(all_test_client_results_)
    all_eval_rounds_ = np.array(all_eval_rounds_)
    
    return all_client_results_, all_test_client_results_, all_eval_rounds_


if __name__ == "__main__":
    print(f"Starting eval_fedavg pipeline...")
    start = time.time()
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    arguments_manager = TrainArgumentsManager()
    arguments_manager.parse_arguments()

    all_client_results, all_test_client_results, all_eval_rounds = run(arguments_manager)
    
    if "results_dir" in arguments_manager.args:
        results_dir = arguments_manager.args.results_dir
    else:
        results_dir = os.path.join("results", arguments_manager.args_to_string())

    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "fedavg_all_eval_rounds.npy"), all_eval_rounds)
    np.save(os.path.join(results_dir, "fedavg_all_client_results.npy"), all_client_results)

    print(f"\tElapsed time for eval_fedavg pipeline = {(time.time() - start):.2f}s")