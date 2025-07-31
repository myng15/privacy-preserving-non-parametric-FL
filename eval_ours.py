"""Adapted from: https://github.com/omarfoq/knn-per/tree/main"""

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from collections import defaultdict
from torch.utils.data import Subset

from utils.utils import *
from utils.args import TestArgumentsManager
from utils.privacy_metrics import *

from geomloss import SamplesLoss

def eval_knn_grid(client_, weights_grid_, capacities_grid_):
    client_accuracies = np.zeros((len(weights_grid_), len(capacities_grid_)))
    client_balanced_accuracies = np.zeros((len(weights_grid_), len(capacities_grid_)))
    client_auc_scores = np.zeros((len(weights_grid_), len(capacities_grid_)))
    client_f1_scores_macro = np.zeros((len(weights_grid_), len(capacities_grid_)))
    client_f1_scores_weighted = np.zeros((len(weights_grid_), len(capacities_grid_)))

    for ii, capacity in enumerate(capacities_grid_):
        client_.capacity = capacity
        client_.clear_datastore()
        client_.build_datastore()
        
        for jj, weight in enumerate(weights_grid_):
            client_acc, client_balanced_acc, client_auc, client_f1_macro, client_f1_weighted = client_.evaluate(weight, val_mode=True) # F1
            client_accuracies[jj, ii] = client_acc * client_.n_val_samples
            client_balanced_accuracies[jj, ii] = client_balanced_acc * client_.n_val_samples
            client_auc_scores[jj, ii] = client_auc * client_.n_val_samples
            client_f1_scores_macro[jj, ii] = client_f1_macro * client_.n_val_samples 
            client_f1_scores_weighted[jj, ii] = client_f1_weighted * client_.n_val_samples 

    return client_accuracies, client_balanced_accuracies, client_auc_scores, client_f1_scores_macro, client_f1_scores_weighted # F1

def eval_knn(client_, weight_, capacity_):
    # Set client to the optimal capacity
    client_.capacity = capacity_
    
    # Clear datastore and rebuild it with the chosen capacity
    client_.clear_datastore()
    client_.build_datastore()
    
    # Evaluate the client with the optimal weight (λ) and fixed capacity
    client_acc, client_balanced_acc, client_auc, client_f1_macro, client_f1_weighted = client_.evaluate(weight_, val_mode=False)
    client_acc = client_acc * client_.n_test_samples
    client_balanced_acc = client_balanced_acc * client_.n_test_samples
    client_auc = client_auc * client_.n_test_samples
    client_f1_macro = client_f1_macro * client_.n_test_samples # F1
    client_f1_weighted = client_f1_weighted * client_.n_test_samples # F1

    return client_acc, client_balanced_acc, client_auc, client_f1_macro, client_f1_weighted

def run(arguments_manager_):

    if not arguments_manager_.initialized:
        arguments_manager_.parse_arguments()

    args_ = arguments_manager_.args
    
    seed_everything(args_.seed)

    rng_seed = args_.seed 
    rng = np.random.default_rng(seed=rng_seed)

    data_dir = get_data_dir(args_.experiment)

    weights_grid_ = np.arange(0., 1. + 1e-6, args_.weights_grid_resolution)
    capacities_grid_ = np.arange(0., 1. + 1e-6, args_.capacities_grid_resolution)

    print("===> Initializing clients...")

    features_dimension = EMBEDDING_DIM[args_.backbone] 

    # FOR DATASETS WITH NATURAL DOMAINS
    if args_.experiment in ["camelyon17", "fitzpatrick17k", "epistroma", "covidfl"]:
        train_loaders, val_loaders, test_loaders = get_loaders(
                experiment_=args_.experiment,
                aggregator_= args_.aggregator_type, 
                data_dir=os.path.join(data_dir, "train"), 
                batch_size=args_.bz,
                is_validation=False,
            )

        num_clients = len(train_loaders)
        clients = []

        for client_id, (train_loader, val_loader, test_loader) in enumerate(tqdm(
            zip(train_loaders, val_loaders, test_loaders), 
            total=num_clients
        )):
        
            if args_.verbose > 0:
                print(f"[Client ID: {client_id}] N_Train: {len(train_loader.dataset)} | N_Val: {len(val_loader.dataset)} | N_Test: {len(test_loader.dataset)}")
        
            client = get_client(
                client_type=args_.client_type, 
                learner=None,
                train_iterator=train_loader,
                val_iterator=val_loader,
                test_iterator=test_loader,
                logger=None,
                client_id=client_id,
                args=args_,
                features_dimension=features_dimension,
                rng=rng
            )

            if client.n_train_samples == 0 or client.n_test_samples == 0:
                continue

            client.load_all_features_and_labels()

            clients.append(client)

    
    # FOR DATASETS WITH ARTIFICIAL CLIENT SPLITS
    else:
        _, train_loaders, test_loaders = get_loaders(
            experiment_=args_.experiment,
            aggregator_= args_.aggregator_type, 
            data_dir=os.path.join(data_dir, "train"), 
            batch_size=args_.bz,
            is_validation=False,
        )

        num_clients = len(train_loaders)
        clients = []

        for client_id, (train_loader, test_loader) in enumerate(tqdm(
            zip(train_loaders, test_loaders), 
            total=num_clients
        )):
        
            # WITH STRATIFIED SPLIT - MANUAL
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
                
            client = get_client(
                client_type=args_.client_type, 
                learner=None,
                train_iterator=train_loader,
                val_iterator=val_loader,
                test_iterator=test_loader,
                logger=None,
                client_id=client_id,
                args=args_,
                features_dimension=features_dimension,
                rng=rng
            )

            if client.n_train_samples == 0 or client.n_test_samples == 0:
                continue
            
            client.load_all_features_and_labels()

            clients.append(client)
    

    aggregator = \
        get_aggregator(
            aggregator_type=args_.aggregator_type, 
            clients=clients,
            features_dimension=features_dimension,
            seed=args_.seed
        )
    
    # PART 1: FEDERATED DATA SHARING + DOWNSTREAM VALIDATION
    print("===> Federated data sharing + Validation starts...")

    # CVAE-FEDAVG
    if args_.anonymizer == "cvae_fedavg":
        chkpts_dir = os.path.join(args_.chkpts_path, "cvae_fedavg")
        os.makedirs(chkpts_dir, exist_ok=True)

        g = torch.Generator()
        g.manual_seed(args_.seed)
        anonymizer = CVAEAnonymizer(args=args_, g=g)
        
        if args_.use_pretrained_cvae_fedavg:
            for client in tqdm(clients): 
                client.trainer = anonymizer.get_trainer(
                    client=client, 
                    is_trained=False)

                best_ckpt = os.path.join(chkpts_dir, f'client_{client.id}.pt') 
                if os.path.isfile(best_ckpt): 
                    client.trainer.load_checkpoint() # load best ckpt
                else:
                    raise FileNotFoundError(f"Client_{client.id}'s checkpoint file not found: {best_ckpt}")

        else:
            # Initialize global CVAE model
            global_trainer = anonymizer.get_trainer(
                client=None, 
                is_trained=False)

            # Train a initial local CVAE for each client
            for client in tqdm(clients): 
                client.trainer = anonymizer.get_trainer(
                    client=client, 
                    is_trained=False)


            cvae_aggregator = CVAEAggregator(
                clients=clients,
                global_trainer=global_trainer,
                anonymizer=anonymizer,
                log_freq=10, 
                lr=args_.anonymizer_lr, 
                verbose=args_.verbose,
                seed=args_.seed
            )

            print("Training CVAE-FedAvg..")
            for ii in tqdm(range(args_.n_fedavg_rounds)): 
                cvae_aggregator.mix()
                
                if (ii % args_.log_freq) == (args_.log_freq - 1):
                    cvae_aggregator.write_logs(anonymizer)

            cvae_aggregator.save_state(chkpts_dir)    



    # CGAN-FEDAVG
    if args_.anonymizer == "cgan_fedavg":
        g = torch.Generator()
        g.manual_seed(args_.seed)
        anonymizer = CGANAnonymizer(args=args_, g=g)

        # Initialize global CGAN model
        global_trainer = anonymizer.get_trainer(
            client=None, 
            is_trained=False)

        # Train a initial local CGAN for each client
        for client in tqdm(clients): 
            client.trainer = anonymizer.get_trainer(
                client=client, 
                is_trained=False)


        cgan_aggregator = CGANAggregator(
            clients=clients,
            global_trainer=global_trainer,
            anonymizer=anonymizer,
            log_freq=args_.log_freq,
            lr=args_.anonymizer_lr, 
            verbose=args_.verbose,
            seed=args_.seed
        )

        print("Training CGAN-FedAvg..")
        chkpts_dir = os.path.join(args_.chkpts_path, "cgan_fedavg")
        os.makedirs(chkpts_dir, exist_ok=True)
        
        for ii in tqdm(range(args_.n_fedavg_rounds)): 
            cgan_aggregator.mix()
            
            if (ii % args_.log_freq) == (args_.log_freq - 1):
                cgan_aggregator.write_logs(anonymizer)

        cgan_aggregator.save_state(chkpts_dir)    


    # Tuning λ and capacity using validation set
    best_weight = 0.0
    best_capacity = 0.0

    all_client_ids_ = []
    all_scores_ = []
    all_balanced_acc_scores_ = []
    all_auc_scores_ = []
    all_f1_scores_macro_ = [] 
    all_f1_scores_weighted_ = [] 
    n_train_samples_ = []
    n_val_samples_ = []
    n_integrated_global_samples_ = []
    all_wasserstein_dist_ = []

    # CENTROID-BASED ANONYMIZATION METHODS (incl. DP-kSame) - Global Aggregation
    if args_.anonymizer == "centroids":
        for client in tqdm(clients):
            features_shared, labels_shared = client.anonymize_data()
            aggregator.get_data_from_client(client.id, features_shared, labels_shared)

        print("===> Aggregator aggregates the anonymized clients' data...")
        aggregator.aggregate_all_clients_data()
        

    # PROTOTYPE-BASED ANONYMIZATION METHODS (incl. DP-kSame) - Global Aggregation
    global_protos = defaultdict(list)
    if args_.anonymizer == "fedproto":
        local_proto_list = []
        for client in tqdm(clients):
            client.local_protos = client.compute_prototypes(client.train_features, client.train_labels)
            local_proto_list.append(client.local_protos)
        
        global_protos = aggregator.aggregate_prototypes(local_proto_list)
        proto_features, proto_labels = aggregator.get_proto_data(global_protos)


    for client in tqdm(clients):
        
        # CENTROID-BASED ANONYMIZATION METHODS (incl. DP-kSame)
        if args_.anonymizer == "centroids":
            if args_.global_sampling == "class_balanced":
                shared_features, shared_labels = aggregator.send_relevant_features_class_balanced(client.id, client.train_labels) 
            elif args_.global_sampling == "random":
                shared_features, shared_labels = aggregator.send_relevant_features(client.id, client.train_labels) 
        # OTHER ANONYMIZATION METHODS
        elif args_.anonymizer is not None and args_.anonymizer != "centroids" and args_.anonymizer != "fedproto": 
            shared_features, shared_labels = client.anonymize_data() 

        elif args_.anonymizer == "fedproto": 
            shared_features, shared_labels = proto_features, proto_labels
            client.global_protos = global_protos

        else:
            shared_features, shared_labels = [], []

        if len(shared_features) > 0:
            # SAVE REAL AND GENERATED EMBEDDINGS FOR LATER EVALUATION
            client_embeddings = {
                'real_embeddings': client.train_features,
                'real_labels': client.train_labels,
                'real_val_embeddings': client.val_features,
                'real_val_labels': client.val_labels,
                'real_test_embeddings': client.test_features,
                'real_test_labels': client.test_labels,
                'generated_embeddings': np.array(shared_features),
                'generated_labels': np.array(shared_labels)
            }

            embeddings_save_path = os.path.join(args_.results_dir, 'saved_embeddings')
            os.makedirs(embeddings_save_path, exist_ok=True)
            np.savez(os.path.join(embeddings_save_path,f'client_{client.id}.npz'), **client_embeddings)

            
            # Compute Wasserstein distance via Sinkhorn divergence
            real_embeddings = torch.tensor(client.train_features, dtype=torch.float32).cuda()
            generated_embeddings = torch.tensor(np.array(shared_features), dtype=torch.float32).cuda()

            sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
            wasserstein_distance = sinkhorn_loss(real_embeddings, generated_embeddings).item()
            wasserstein_distance = np.sqrt(wasserstein_distance)
            all_wasserstein_dist_.append(wasserstein_distance)

        relevant_features, relevant_labels = shared_features, np.array(shared_labels)

        all_client_ids_.append(client.id)
        if len(relevant_features) > 0:
            n_integrated_global_samples_.append(len(relevant_features))
            client.integrate_global_data(relevant_features, relevant_labels) 


        # VALIDATION (Grid search)
        client_scores, client_balanced_acc_scores, client_auc_scores, client_f1_scores_macro, client_f1_scores_weighted = eval_knn_grid(client, weights_grid_, capacities_grid_) 
        n_train_samples_.append(client.n_train_samples)
        n_val_samples_.append(client.n_val_samples)
        all_scores_.append(client_scores)
        all_balanced_acc_scores_.append(client_balanced_acc_scores)
        all_auc_scores_.append(client_auc_scores)
        all_f1_scores_macro_.append(client_f1_scores_macro) 
        all_f1_scores_weighted_.append(client_f1_scores_weighted) 


    all_client_ids_ = np.array(all_client_ids_)
    all_scores_ = np.array(all_scores_)
    all_balanced_acc_scores_ = np.array(all_balanced_acc_scores_)
    all_auc_scores_ = np.array(all_auc_scores_)
    all_f1_scores_macro_ = np.array(all_f1_scores_macro_)
    all_f1_scores_weighted_ = np.array(all_f1_scores_weighted_)
    n_train_samples_ = np.array(n_train_samples_)
    n_val_samples_ = np.array(n_val_samples_)
    n_integrated_global_samples_ = np.array(n_integrated_global_samples_)
    all_wasserstein_dist_ = np.array(all_wasserstein_dist_)
    
    avg_wasserstein_dist = np.mean(all_wasserstein_dist_)
    print(f"Overall average Wasserstein distance across clients: {avg_wasserstein_dist:.6f}")
    
    # Calculate average test accuracy (across all clients) for each combination of weight and capacity and find the best combo
    normalized_scores = np.nan_to_num(all_scores_) / n_val_samples_[:, np.newaxis, np.newaxis]
    accuracies = normalized_scores.mean(axis=0) * 100

    best_acc = np.max(accuracies)
    print(f"best_acc from eval_knn_grid(): {best_acc} ~approx. {best_acc:.2f}")
    best_index = np.unravel_index(np.argmax(accuracies), accuracies.shape)
    best_weight = weights_grid_[best_index[0]]
    best_capacity = capacities_grid_[best_index[1]]
    print(f"Optimal weight: {best_weight:.2f}, Optimal Capacity: {best_capacity:.2f}")

    # Calculate average balanced test accuracy (across all clients) for each combination of weight and capacity and find the best combo
    normalized_scores = np.nan_to_num(all_balanced_acc_scores_) / n_val_samples_[:, np.newaxis, np.newaxis]
    balanced_accuracies = normalized_scores.mean(axis=0) * 100

    best_balanced_acc = np.max(balanced_accuracies)
    print(f"best_balanced_acc from eval_knn_grid(): {best_balanced_acc} ~approx. {best_balanced_acc:.2f}")
    best_index_balanced = np.unravel_index(np.argmax(balanced_accuracies), balanced_accuracies.shape)
    best_weight_balanced = weights_grid_[best_index_balanced[0]]
    best_capacity_balanced = capacities_grid_[best_index_balanced[1]]
    print(f"Optimal weight for Balanced Accuracy: {best_weight_balanced:.2f}, Optimal Capacity for Balanced Accuracy: {best_capacity_balanced:.2f}")

    # Calculate average ROC-AUC scores (across all clients) for each combination of weight and capacity and find the best combo
    auc_scores = (np.nan_to_num(all_auc_scores_).sum(axis=0) / n_val_samples_.sum())
    best_auc = np.max(auc_scores)
    print(f"best_auc from eval_knn_grid(): {best_auc} ~approx. {best_auc:.3f}")
    best_index_auc = np.unravel_index(np.argmax(auc_scores), auc_scores.shape)
    best_weight_auc = weights_grid_[best_index_auc[0]]
    best_capacity_auc = capacities_grid_[best_index_auc[1]]
    print(f"Optimal weight for ROC-AUC: {best_weight_auc:.2f}, Optimal Capacity for ROC-AUC: {best_capacity_auc:.2f}")

    # Calculate average F1 scores (Macro) (across all clients) for each combination of weight and capacity and find the best combo
    f1_scores_macro = (np.nan_to_num(all_f1_scores_macro_).sum(axis=0) / n_val_samples_.sum())
    best_f1_macro = np.max(f1_scores_macro)
    print(f"best_f1_macro from eval_knn_grid(): {best_f1_macro} ~approx. {best_f1_macro:.3f}")
    best_index_f1_macro = np.unravel_index(np.argmax(f1_scores_macro), f1_scores_macro.shape)
    best_weight_f1_macro = weights_grid_[best_index_f1_macro[0]]
    best_capacity_f1_macro = capacities_grid_[best_index_f1_macro[1]]
    print(f"Optimal weight for F1 score (Macro): {best_weight_f1_macro:.2f}, Optimal Capacity for F1 score (Macro): {best_capacity_f1_macro:.2f}")

    # Calculate average F1 scores (Weighted) (across all clients) for each combination of weight and capacity and find the best combo
    f1_scores_weighted = (np.nan_to_num(all_f1_scores_weighted_).sum(axis=0) / n_val_samples_.sum())
    best_f1_weighted = np.max(f1_scores_weighted)
    print(f"best_f1_mweighted from eval_knn_grid(): {best_f1_weighted} ~approx. {best_f1_weighted:.3f}")
    best_index_f1_weighted = np.unravel_index(np.argmax(f1_scores_weighted), f1_scores_weighted.shape)
    best_weight_f1_weighted = weights_grid_[best_index_f1_weighted[0]]
    best_capacity_f1_weighted = capacities_grid_[best_index_f1_weighted[1]]
    print(f"Optimal weight for F1 score (Weighted): {best_weight_f1_weighted:.2f}, Optimal Capacity for F1 score (Weighted): {best_capacity_f1_weighted:.2f}")



    # PART 2: EVALUATE ON TEST SET USING OPTIMAL HYPERPARAMETERS
    print("===> Evaluation on test set using optimal weight + optimal capacity starts...")

    all_test_scores_ = []
    all_test_balanced_acc_scores_ = []
    all_test_auc_scores_ = []
    all_test_f1_scores_macro_ = [] 
    all_test_f1_scores_weighted_ = [] 
    n_test_samples_ = []
    
    for client in tqdm(clients):
        test_score, test_balanced_acc_score, test_auc, test_f1_macro, test_f1_weighted = eval_knn(client, best_weight, best_capacity) # F1
        all_test_scores_.append(test_score)
        all_test_balanced_acc_scores_.append(test_balanced_acc_score)
        all_test_auc_scores_.append(test_auc)
        all_test_f1_scores_macro_.append(test_f1_macro)
        all_test_f1_scores_weighted_.append(test_f1_weighted) 
        n_test_samples_.append(client.n_test_samples)
    
    all_test_scores_ = np.array(all_test_scores_) 
    all_test_balanced_acc_scores_ = np.array(all_test_balanced_acc_scores_) 
    all_test_auc_scores_ = np.array(all_test_auc_scores_) 
    all_test_f1_scores_macro_ = np.array(all_test_f1_scores_macro_) 
    all_test_f1_scores_weighted_ = np.array(all_test_f1_scores_weighted_) 
    n_test_samples_ = np.array(n_test_samples_)

    average_test_accuracy = np.sum(all_test_scores_) / np.sum(n_test_samples_) * 100
    individual_accuracies = (all_test_scores_ / n_test_samples_) * 100
    mean_test_acc = np.mean(individual_accuracies)
    std_test_acc = np.std(individual_accuracies)
    print(f'Average Test Accuracy: {average_test_accuracy:.2f}%')
    print(f'Mean and Standard Deviation of Test Accuracies across Clients: Mean: {mean_test_acc:.2f}%, Std: {std_test_acc:.2f}%')

    average_test_balanced_accuracy = np.sum(all_test_balanced_acc_scores_) / np.sum(n_test_samples_) * 100
    individual_balanced_accuracies = (average_test_balanced_accuracy / n_test_samples_) * 100
    mean_test_balanced_acc = np.mean(individual_balanced_accuracies)
    std_test_balanced_acc = np.std(individual_balanced_accuracies)
    print(f'Average Test Balanced Accuracy: {average_test_balanced_accuracy:.2f}%')
    print(f'Mean and Standard Deviation of Test Balanced Accuracies across Clients: Mean: {mean_test_balanced_acc:.2f}%, Std: {std_test_balanced_acc:.2f}%')
    
    average_test_auc = np.sum(all_test_auc_scores_) / np.sum(n_test_samples_)
    individual_auc = (all_test_auc_scores_ / n_test_samples_)
    mean_test_auc = np.mean(individual_auc)
    std_test_auc = np.std(individual_auc)
    print(f'Average Test ROC-AUC: {average_test_auc:.3f}')
    print(f'Mean and Standard Deviation of Test ROC-AUC across Clients: Mean: {mean_test_auc:.3f}, Std: {std_test_auc:.3f}')


    average_test_f1_macro = np.sum(all_test_f1_scores_macro_) / np.sum(n_test_samples_)
    individual_f1_macro = (all_test_f1_scores_macro_ / n_test_samples_)
    mean_test_f1_macro = np.mean(individual_f1_macro)
    std_test_f1_macro = np.std(individual_f1_macro)
    print(f'Average Test F1 Score (Macro): {average_test_f1_macro:.3f}')
    print(f'Mean and Standard Deviation of Test F1 Score (Macro) across Clients: Mean: {mean_test_f1_macro:.3f}, Std: {std_test_f1_macro:.3f}')

    average_test_f1_weighted = np.sum(all_test_f1_scores_weighted_) / np.sum(n_test_samples_)
    individual_f1_weighted = (all_test_f1_scores_weighted_ / n_test_samples_)
    mean_test_f1_weighted = np.mean(individual_f1_weighted)
    std_test_f1_weighted = np.std(individual_f1_weighted)
    print(f'Average Test F1 Score (Weighted): {average_test_f1_weighted:.3f}')
    print(f'Mean and Standard Deviation of Test F1 Score (Weighted) across Clients: Mean: {mean_test_f1_weighted:.3f}, Std: {std_test_f1_weighted:.3f}')


    return {
        'all_client_ids': all_client_ids_,
        'weights_grid': weights_grid_, 
        'capacities_grid': capacities_grid_, 
        'n_integrated_global_samples': n_integrated_global_samples_, 
        'all_scores': all_scores_, 
        'all_balanced_acc_scores': all_balanced_acc_scores_, 
        'all_auc_scores': all_auc_scores_, 
        'all_f1_scores_macro': all_f1_scores_macro_, 
        'all_f1_scores_weighted': all_f1_scores_weighted_, 
        'n_train_samples': n_train_samples_, 
        'n_val_samples': n_val_samples_,
        'best_acc': best_acc, 
        'best_weight': best_weight, 
        'best_balanced_acc': best_balanced_acc, 
        'best_weight_balanced': best_weight_balanced, 
        'best_auc': best_auc, 
        'best_weight_auc': best_weight_auc,  
        'best_f1_macro': best_f1_macro, 
        'best_weight_f1_macro': best_weight_f1_macro, 
        'best_f1_weighted': best_f1_weighted, 
        'best_weight_f1_weighted': best_weight_f1_weighted, 
        'all_test_scores': all_test_scores_, 
        'all_test_balanced_acc_scores': all_test_balanced_acc_scores_, 
        'all_test_auc_scores': all_test_auc_scores_, 
        'all_test_f1_scores_macro': all_test_f1_scores_macro_, 
        'all_test_f1_scores_weighted': all_test_f1_scores_weighted_, 
        'n_test_samples': n_test_samples_, 
        'average_test_accuracy': average_test_accuracy, 
        'mean_test_acc': mean_test_acc, 
        'std_test_acc': std_test_acc, 
        'average_test_balanced_accuracy': average_test_balanced_accuracy, 
        'mean_test_balanced_acc': mean_test_balanced_acc, 
        'std_test_balanced_acc': std_test_balanced_acc, 
        'average_test_auc': average_test_auc, 
        'mean_test_auc': mean_test_auc, 
        'std_test_auc': std_test_auc,
        'average_test_f1_macro': average_test_f1_macro, 
        'mean_test_f1_macro': mean_test_f1_macro, 
        'std_test_f1_macro': std_test_f1_macro, 
        'average_test_f1_weighted': average_test_f1_weighted,
        'mean_test_f1_weighted': mean_test_f1_weighted, 
        'std_test_f1_weighted': std_test_f1_weighted, 
        'all_wasserstein_dist': all_wasserstein_dist_
    }

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    arguments_manager = TestArgumentsManager()
    arguments_manager.parse_arguments()

    if "results_dir" in arguments_manager.args:
        results_dir = arguments_manager.args.results_dir
    else:
        results_dir = os.path.join("results", arguments_manager.args_to_string())

    os.makedirs(results_dir, exist_ok=True)

    results_dict = run(arguments_manager) 
    with open(os.path.join(results_dir, "results_dict.pkl"), 'wb') as f:
        pickle.dump(results_dict, f)

    # Save the results to a seed-specific log file
    with open(os.path.join(results_dir, "results.log"), "w") as log_file:
        log_file.write(f"Best Accuracy in Grid-Search evaluation: {results_dict['best_acc']:.2f}\n")
        log_file.write(f"Best Balanced Accuracy in Grid-Search evaluation: {results_dict['best_balanced_acc']:.2f}\n")
        log_file.write(f"Best ROC-AUC in Grid-Search evaluation: {results_dict['best_auc']:.3f}\n")
        log_file.write(f"Best F1 Score (Macro) in Grid-Search evaluation: {results_dict['best_f1_macro']:.3f}\n") 
        log_file.write(f"Best F1 Score (Weighted) in Grid-Search evaluation: {results_dict['best_f1_weighted']:.3f}\n") 
        log_file.write(f"Optimal weight: {results_dict['best_weight']:.2f}\n")
        log_file.write(f"Optimal weight for Balanced Accuracy: {results_dict['best_weight_balanced']:.2f}\n")
        log_file.write(f"Optimal weight for ROC-AUC: {results_dict['best_weight_auc']:.2f}\n")
        log_file.write(f"Optimal weight for F1 Score (Macro): {results_dict['best_weight_f1_macro']:.2f}\n") 
        log_file.write(f"Optimal weight for F1 Score (Weighted): {results_dict['best_weight_f1_weighted']:.2f}\n") 
        log_file.write(f"Test Accuracy using Optimal weight (Average | Mean | Std): {results_dict['average_test_accuracy']:.2f} | {results_dict['mean_test_acc']:.2f} | {results_dict['std_test_acc']:.2f}\n")
        log_file.write(f"Test Balanced Accuracy using Optimal weight (Average | Mean | Std): {results_dict['average_test_balanced_accuracy']:.2f} | {results_dict['mean_test_balanced_acc']:.2f} | {results_dict['std_test_balanced_acc']:.2f}\n")
        log_file.write(f"Test ROC-AUC using Optimal weight (Average | Mean | Std): {results_dict['average_test_auc']:.2f} | {results_dict['mean_test_auc']:.2f} | {results_dict['std_test_auc']:.2f}\n")
        log_file.write(f"Test F1 Score (Macro) using Optimal weight (Average | Mean | Std): {results_dict['average_test_f1_macro']:.2f} | {results_dict['mean_test_f1_macro']:.2f} | {results_dict['std_test_f1_macro']:.2f}\n") # F1
        log_file.write(f"Test F1 Score (Weighted) using Optimal weight (Average | Mean | Std): {results_dict['average_test_f1_weighted']:.2f} | {results_dict['mean_test_f1_weighted']:.2f} | {results_dict['std_test_f1_weighted']:.2f}\n") # F1

    