from collections import Counter, defaultdict
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

from anonymizer.centroids import CentroidAnonymizer
from datastore import *
from utils.optim import get_lr_scheduler, get_optimizer
from utils.torch_utils import *
from utils.constants import *

from copy import deepcopy
import time

from faiss import IndexFlatL2

import chromadb
from chromadb.config import Settings
from uuid import uuid4

from sklearn.cluster import KMeans
from tqdm import tqdm

from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from scipy.special import softmax

from kmeans_pytorch import kmeans

from anonymizer.ksame import kSame
from anonymizer.cvae import CVAEAnonymizer
from anonymizer.cwae import CWAEAnonymizer
from anonymizer.cgan import CGANAnonymizer

from models.classification_models import LinearLayer, MultiLayerPerceptron

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Client(object):
    r"""
    Implements one client
    Adapted from: https://github.com/omarfoq/knn-per/tree/main
    """
    def __init__(
            self,
            learner,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            save_path=None,
            id_=None,
            *args,
            **kwargs
    ):
        self.learner = learner

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator

        self.n_train_samples = len(self.train_iterator.dataset)
        self.n_val_samples = len(self.val_iterator.dataset)
        self.n_test_samples = len(self.test_iterator.dataset)

        self.local_steps = local_steps

        self.save_path = save_path

        self.id = -1
        if id_ is not None:
            self.id = id_

        self.counter = 0
        self.logger = logger

    def is_ready(self):
        return self.learner.is_ready

    def step(self, *args, **kwargs):
        self.counter += 1

        self.learner.fit_epochs(
            iterator=self.train_iterator,
            n_epochs=self.local_steps,
        )

    def write_logs(self):
        train_loss, train_acc, train_balanced_acc, train_auc = self.learner.evaluate_iterator(self.val_iterator)
        test_loss, test_acc, test_balanced_acc, test_auc = self.learner.evaluate_iterator(self.test_iterator)

        self.logger.add_scalar("Train/Loss", train_loss, self.counter)
        self.logger.add_scalar("Train/Metric", train_acc, self.counter)
        self.logger.add_scalar("Train/Balanced Accuracy", train_balanced_acc, self.counter)
        self.logger.add_scalar("Train/ROC-AUC", train_auc, self.counter)
        self.logger.add_scalar("Test/Loss", test_loss, self.counter)
        self.logger.add_scalar("Test/Metric", test_acc, self.counter)
        self.logger.add_scalar("Test/Balanced Accuracy", test_balanced_acc, self.counter)
        self.logger.add_scalar("Test/ROC-AUC", test_auc, self.counter)

        return train_loss, train_acc, train_balanced_acc, train_auc, test_loss, test_acc, test_balanced_acc, test_auc

    def save_state(self, path=None):
        """
        :param path: expected to be a `.pt` file
        """
        if path is None:
            if self.save_path is None:
                warnings.warn("client state was not saved", RuntimeWarning)
                return
            else:
                self.learner.save_checkpoint(self.save_path)
                return

        self.learner.save_checkpoint(path)

    def load_state(self, path=None):
        if path is None:
            if self.save_path is None:
                warnings.warn("client state was not loaded", RuntimeWarning)
                return
            else:
                self.learner.load_checkpoint(self.save_path)
                return

        self.learner.load_checkpoint(path)

    def free_memory(self):
        self.learner.free_memory()


class KNNClient(Client):
    def __init__(
            self, 
            learner,
            train_iterator, 
            val_iterator,
            test_iterator, 
            logger, 
            id_,
            args_,
            k,
            n_clusters, 
            features_dimension, 
            num_classes,
            capacity, 
            strategy, 
            rng, 
            knn_weights,
            gaussian_kernel_scale,
            device,
            seed=1234,
            *args, 
            **kwargs
    ):
        super(KNNClient, self).__init__(
            learner=learner,
            train_iterator=train_iterator,
            val_iterator=val_iterator,  
            test_iterator=test_iterator,
            logger=logger,
            id_=id_,
            local_steps=None,  
            *args, 
            **kwargs
        )

        self.args_ = args_

        self.k = k
        self.n_clusters = n_clusters
        self.knn_weights = knn_weights
        self.gaussian_kernel_scale = gaussian_kernel_scale

        self.features_dimension = features_dimension
        self.num_classes = num_classes

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator

        self.n_train_samples = len(train_iterator.dataset)
        self.n_val_samples = len(val_iterator.dataset)
        self.n_test_samples = len(test_iterator.dataset)

        self.capacity = capacity
        self.strategy = strategy
        self.rng = rng

        # Initialize FAISS index for global features
        self.faiss_index = IndexFlatL2(self.features_dimension)
        self.global_features = np.array([], dtype=np.float32)
        self.global_labels = np.array([], dtype=np.int64)
        
        # Initialize local datastore for embeddings and labels
        self.datastore = DataStore(self.capacity, self.strategy, self.features_dimension, self.rng)
        self.datastore_flag = False

        # Initialize ChromaDB for cosine similarity search
        self.use_cosine_knn = self.args_.knn_metric == "cosine"
        if self.use_cosine_knn:
            self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
            self.chroma_local_collection = None
            self.chroma_global_collection = None

        # Initialize local training data
        self.train_features = np.zeros(shape=(self.n_train_samples, self.features_dimension), dtype=np.float32)
        self.train_labels = np.zeros(shape=self.n_train_samples, dtype=np.int64)
        self.val_features = np.zeros(shape=(self.n_val_samples, self.features_dimension), dtype=np.float32)
        self.val_labels = np.zeros(shape=self.n_val_samples, dtype=np.int64)
        self.test_features = np.zeros(shape=(self.n_test_samples, self.features_dimension), dtype=np.float32)
        self.test_labels = np.zeros(shape=self.n_test_samples, dtype=np.int64)

        self.local_knn_outputs = np.zeros(shape=(self.n_test_samples, self.num_classes), dtype=np.float32)
        self.local_knn_outputs_flag = False

        self.glocal_knn_outputs = np.zeros(shape=(self.n_test_samples, self.num_classes), dtype=np.float32)
        self.glocal_knn_outputs_flag = False

        self.device = device 
        self.seed = seed

        # CVAE-FEDAVG
        self.trainer = None

        # FedProto
        self.global_protos = defaultdict(list)


    @property
    def k(self):
        return self.__k

    @k.setter
    def k(self, k):
        self.__k = int(k)

    @property
    def capacity(self):
        return self.__capacity

    @capacity.setter
    def capacity(self, capacity):
        if 0 <= capacity <= 1 and isinstance(capacity, float):
            capacity = int(capacity * self.n_train_samples)
        else:
            capacity = int(capacity)

        if capacity < 0:
            capacity = self.n_train_samples

        self.__capacity = capacity

    def _seed_everything_client(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_all_features_and_labels(self):
        # Load train data
        train_features_list = []
        train_labels_list = []
        
        for batch_features, batch_labels, _ in self.train_iterator:
            train_features_list.append(batch_features.numpy())
            train_labels_list.append(batch_labels.numpy())

        self.train_features = np.concatenate(train_features_list, axis=0)
        self.train_labels = np.concatenate(train_labels_list, axis=0)

        # Load validation data
        val_features_list = []
        val_labels_list = []

        for batch_features, batch_labels, _ in self.val_iterator:
            val_features_list.append(batch_features.numpy())
            val_labels_list.append(batch_labels.numpy())

        self.val_features = np.concatenate(val_features_list, axis=0) 
        self.val_labels = np.concatenate(val_labels_list, axis=0)

        # Load test data
        test_features_list = []
        test_labels_list = []

        for batch_features, batch_labels, _ in self.test_iterator:
            test_features_list.append(batch_features.numpy())
            test_labels_list.append(batch_labels.numpy())

        self.test_features = np.concatenate(test_features_list, axis=0)
        self.test_labels = np.concatenate(test_labels_list, axis=0)

    def build_datastore(self):
        self.datastore_flag = True

        # CHROMA (for Cosine similarity)
        if self.use_cosine_knn:
            self.chroma_local_collection = self.chroma_client.get_or_create_collection(
                name=f"local_{uuid4().hex}", metadata={"hnsw:space": "cosine"}
            )
            self._populate_chroma(self.chroma_local_collection, self.train_features, self.train_labels)
        else: # FAISS (for Euclidean distance)
            self.datastore.build(self.train_features, self.train_labels)


    def compute_prototypes(self, features, labels):
        protos = defaultdict(list)

        # Step 1: group features by class
        for feat, label in zip(features, labels):
            feat = np.asarray(feat)
            protos[int(label)].append(feat)
        
        # Step 2: compute prototype for each class
        for label in protos:
            feats = np.stack(protos[label])  
            proto = feats.mean(axis=0)
            protos[label] = proto

        return protos


    def anonymize_data(self, train_features=None, train_labels=None, val_features=None, val_labels=None):
        self._seed_everything_client()

        print(f"Starting anonymization...")
        start = time.time()
        
        train_features = train_features if train_features is not None else self.train_features
        train_labels = train_labels if train_labels is not None else self.train_labels
        val_features = val_features if val_features is not None else self.val_features
        val_labels = val_labels if val_labels is not None else self.val_labels
        
        if self.args_.anonymizer == "centroids": # Centroid-based Anonymization (incl. DP-kSame)
            anonymizer = CentroidAnonymizer(args=self.args_)
            anonymized_features, anonymized_labels = anonymizer.apply(train_features, train_labels)

        elif self.args_.anonymizer.startswith("cvae"): # CVAE-FEDAVG
            g = torch.Generator()
            g.manual_seed(self.seed)
            anonymizer = CVAEAnonymizer(args=self.args_, g=g)
            anonymized_features, anonymized_labels = anonymizer.apply(client=self)
        
        elif self.args_.anonymizer.startswith("cgan"): # CGAN-FEDAVG
            g = torch.Generator()
            g.manual_seed(self.seed)
            anonymizer = CGANAnonymizer(args=self.args_, g=g)
            anonymized_features, anonymized_labels = anonymizer.apply(self)

        print(f"\tElapsed time = {(time.time() - start):.2f}s")
        return anonymized_features, anonymized_labels

    def integrate_global_data(self, global_features, global_labels): 
        #CHROMA (for Cosine similarity)
        if self.use_cosine_knn:
            self.chroma_global_collection = self.chroma_client.get_or_create_collection(
                name=f"global_{uuid4().hex}", metadata={"hnsw:space": "cosine"}
            )
            self._populate_chroma(self.chroma_global_collection, global_features, global_labels)

        else: # FAISS (for Euclidean distance)
            self.faiss_index = faiss.IndexFlatL2(self.features_dimension)
            self.faiss_index.add(global_features)
        
        self.global_features = global_features
        self.global_labels = global_labels

    def _populate_chroma(self, collection, features, labels, batch_size=5000):
        documents = ["sample" for _ in labels]
        metadatas = [{"label": int(label)} for label in labels]
        ids = [str(uuid4()) for _ in labels]
        
        # Chunk the data, since ChromaDB only allows batches of up to 5461 at a time
        for i in range(0, len(labels), batch_size):
            batch_embeddings = features[i:i + batch_size].tolist()
            batch_metadatas = metadatas[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            collection.upsert(
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents,
                ids=batch_ids
            )


    def compute_knn_outputs(self, features, scope="local", method="gaussian_kernel"): 
        """
        Computes k-NN outputs for given features.

        :param features: Features for which to compute k-NN outputs
        :param mode: 'local' or 'global', determines which datastore to use
        :param inverse_distances: If True, use inverse distances as weights
        :param gaussian_kernel: If True, use Gaussian kernel as weights
        :return: k-NN outputs
        """
        # CHROMA (for Cosine similarity)
        if self.use_cosine_knn:
            if scope == "local":
                collection = self.chroma_local_collection
            elif scope == "global":
                collection = self.chroma_global_collection
            else:
                raise ValueError("Scope must be 'local' or 'global'.")

            knn_outputs = self._compute_chroma_outputs(collection, features, method)
            if scope == "local" and knn_outputs is not None:
                self.local_knn_outputs_flag = True
            elif scope == "global" and knn_outputs is not None:
                self.global_knn_outputs_flag = True
        
        else: # FAISS (for Euclidean distance)
            if scope == "local":
                if self.capacity <= 0:
                    return
                assert self.datastore_flag, "Should build local datastore before computing knn outputs!"

                distances, indices = self.datastore.index.search(features, self.k)
                
                if method == "inverse_distances":
                    knn_outputs = self._compute_weighted_outputs(distances, indices)
                elif method == "gaussian_kernel":
                    knn_outputs = self._compute_gaussian_kernel_outputs(distances, indices, self.gaussian_kernel_scale)
                else:
                    raise ValueError("Invalid method. Use 'inverse_distances' or 'gaussian_kernel'.")
                
                if knn_outputs is not None:
                    self.local_knn_outputs_flag = True

            elif scope == "global":
                distances, indices = self.faiss_index.search(features, self.k)

                if method=="inverse_distances":
                    knn_outputs = self._compute_weighted_outputs(distances, indices, global_mode=True)
                elif method=="gaussian_kernel":
                    knn_outputs = self._compute_gaussian_kernel_outputs(distances, indices, self.gaussian_kernel_scale, global_mode=True)
                else:
                    raise ValueError("Invalid method. Use 'inverse_distances' or 'gaussian_kernel'.")

                if knn_outputs is not None:
                    self.global_knn_outputs_flag = True

            else:
                raise ValueError("Scope must be 'local' or 'global'.")

        return knn_outputs

    def _compute_chroma_outputs(self, collection, features, method):
        knn_outputs = np.zeros((len(features), self.num_classes), dtype=np.float32)
        for i, feature in enumerate(features):
            result = collection.query(query_embeddings=feature.tolist(), n_results=self.k, include=["metadatas", "documents", "distances"])
            distances = np.array(result['distances'][0])
            labels = [int(meta['label']) for meta in result['metadatas'][0]]
            
            if method == "inverse_distances":
                weights = 1 / (distances + 1e-8) 
            elif method == "gaussian_kernel":
                weights = np.exp(-distances / (self.features_dimension * self.gaussian_kernel_scale))
            else:
                raise ValueError("Invalid method")

            for weight, label in zip(weights, labels):
                knn_outputs[i, label] += weight
            knn_outputs[i] /= knn_outputs[i].sum() + 1e-8
        return knn_outputs

    def _compute_weighted_outputs(self, distances, indices, global_mode=False):
        if global_mode:
            labels = self.global_labels
        else:
            labels = self.datastore.labels

        if len(labels) < 1:
            return None
        
        # weights of each nearest neighbor in the train dataset to a certain test feature (w.r.t. their distances from this test feature)
        weights = 1. / (distances + 1e-8)  

        knn_outputs = np.zeros((weights.shape[0], self.num_classes), dtype=np.float32) #knn_outputs: (n_test_samples,n_classes); weights: (n_test_samples, k)
        for i in range(weights.shape[0]): # for each test feature
            weighted_sum = np.zeros(self.num_classes, dtype=np.float32)

            for j in range(weights.shape[1]): # for each neighbor of the test feature
                class_label = labels[indices[i, j]]
                weighted_sum[class_label] += weights[i, j]
            knn_outputs[i] = weighted_sum / weights[i].sum()

        return knn_outputs

    def _compute_gaussian_kernel_outputs(self, distances, indices, scale, global_mode=False):
        if global_mode:
            labels = self.global_labels
        else:
            labels = self.datastore.labels

        if len(labels) < 1:
            return None
        
        similarities = np.exp(-distances / (self.features_dimension * scale))
        neighbors_labels = labels[indices]
        knn_outputs = np.zeros((similarities.shape[0], self.num_classes), dtype=np.float32)

        masks = np.zeros(((self.num_classes,) + similarities.shape), dtype=np.float32)
        for class_id in range(self.num_classes):
            masks[class_id] = neighbors_labels == class_id

        knn_outputs = (similarities * masks).sum(axis=2) / similarities.sum(axis=1)

        return knn_outputs.T

    def train_fedproto_classifier(self, model, ckpt_path, epochs=100, alpha=0.5):
        device = self.args_.device
        model.to(device)
        model.train()

        self.local_protos = {c: torch.tensor(p, dtype=torch.float32, device=device)
                     for c, p in self.local_protos.items()}
        self.global_protos = {c: torch.tensor(p, dtype=torch.float32, device=device)
                            for c, p in self.global_protos.items()}

        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(optimizer_name=self.args_.classifier_optimizer, 
                                  model=model, 
                                  lr_initial=self.args_.anonymizer_lr,
                                  weight_decay=0) 

        train_data = torch.tensor(self.train_features, dtype=torch.float32).to(device)
        train_labels = torch.tensor(self.train_labels, dtype=torch.long).to(device)


        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = model(train_data)
            loss_cls = criterion(outputs, train_labels)

            # Regularization loss with global prototypes
            loss_proto = 0.0
            for c in self.local_protos:
                if c in self.global_protos:
                    loss_proto += F.mse_loss(self.local_protos[c], self.global_protos[c])
            loss = loss_cls + alpha * loss_proto
            loss.backward()
            optimizer.step()

        return model


    def train_linear_classifier(self, scope, model, ckpt_path, epochs=100, print_every=10): 
        if scope == "local":
            train_data, train_labels = self.train_features, self.train_labels

        elif scope == "global":
            train_data, train_labels = self.global_features, self.global_labels

        device = self.args_.device
        model.to(device)

        optimizer = get_optimizer(optimizer_name=self.args_.classifier_optimizer, 
                                  model=model, 
                                  lr_initial=self.args_.anonymizer_lr,
                                  weight_decay=5e-4) 

        criterion = nn.CrossEntropyLoss()
        train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
        train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)

        best_val_loss = float('inf')
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            outputs = model(train_data)
            train_loss = criterion(outputs, train_labels)
            train_loss.backward()
            optimizer.step()
            
        torch.save(model.state_dict(), ckpt_path)
        
        return model
    
    def compute_linear_outputs(self, features, scope="local"):
        device = self.args_.device
        features = torch.tensor(features, dtype=torch.float32).to(device)

        if scope == "local":
            if self.capacity <= 0:
                return

            self.local_classifier.to(device) 
            self.local_classifier.eval()
            with torch.no_grad():
                linear_outputs = self.local_classifier(features).cpu().numpy()

            if linear_outputs is not None:
                self.local_knn_outputs_flag = True

        elif scope == "global":
            self.global_classifier.to(device) 
            self.global_classifier.eval()
            with torch.no_grad():
                linear_outputs = self.global_classifier(features).cpu().numpy()

            if linear_outputs is not None:
                self.global_knn_outputs_flag = True
        
        else:
            raise ValueError("Scope must be 'local' or 'global'.")

        return linear_outputs


    def evaluate(self, weight, val_mode):
        """
        Evaluates the client for a given weight parameter.

        :param weight: float in [0, 1]
        :return: accuracy score
        """
        features = self.val_features if val_mode else self.test_features
        labels = self.val_labels if val_mode else self.test_labels 

        if val_mode:
            self.local_knn_outputs = np.zeros(shape=(self.n_val_samples, self.num_classes), dtype=np.float32)
            self.global_knn_outputs = np.zeros(shape=(self.n_val_samples, self.num_classes), dtype=np.float32)
        
        if self.args_.classifier.startswith('fedproto'):
            if self.args_.classifier == 'fedproto_linear':
                self.local_classifier = LinearLayer(self.features_dimension, self.num_classes) 
            elif self.args_.classifier == 'fedproto_mlp':
                self.local_classifier = MultiLayerPerceptron(self.features_dimension, self.num_classes)

            ckpt_path = os.path.join(self.args_.chkpts_path, 'local_classifier')
            os.makedirs(ckpt_path, exist_ok=True)
            client_ckpt = os.path.join(ckpt_path, f'client_{self.id}.pt')

            if not os.path.isfile(client_ckpt): 
                self.local_classifier = self.train_fedproto_classifier(model=self.local_classifier, ckpt_path=client_ckpt, epochs=self.args_.local_epochs)
                
            self.local_classifier.to(self.args_.device) 
            self.local_classifier.eval() 
            features = torch.tensor(features, dtype=torch.float32).to(self.args_.device)
            with torch.no_grad():
                outputs = self.local_classifier(features).cpu().numpy()

        else:
            if self.args_.classifier == 'knn':
                self.local_knn_outputs = self.compute_knn_outputs(features, scope="local", method=self.knn_weights)
                self.global_knn_outputs = self.compute_knn_outputs(features, scope="global", method=self.knn_weights)

            elif self.args_.classifier == 'linear' or self.args_.classifier == 'mlp':
                if self.args_.classifier == 'linear':
                    self.local_classifier = LinearLayer(self.features_dimension, self.num_classes) 
                    self.global_classifier = LinearLayer(self.features_dimension, self.num_classes)
                elif self.args_.classifier == 'mlp':
                    self.local_classifier = MultiLayerPerceptron(self.features_dimension, self.num_classes)
                    self.global_classifier = MultiLayerPerceptron(self.features_dimension, self.num_classes)
                
                ckpt_path = os.path.join(self.args_.chkpts_path, 'local_classifier')
                os.makedirs(ckpt_path, exist_ok=True)
                client_ckpt = os.path.join(ckpt_path, f'client_{self.id}.pt')
                client_ckpt_global = os.path.join(ckpt_path, f'client_{self.id}_global.pt')
                    
                if not os.path.isfile(client_ckpt) or not os.path.isfile(client_ckpt_global): 
                    self.local_classifier = self.train_linear_classifier(scope="local", model=self.local_classifier, ckpt_path=client_ckpt, epochs=self.args_.local_epochs)
                    self.global_classifier = self.train_linear_classifier(scope="global", model=self.global_classifier, ckpt_path=client_ckpt_global, epochs=self.args_.local_epochs)

                self.local_classifier.load_state_dict(torch.load(client_ckpt, weights_only=True))
                self.global_classifier.load_state_dict(torch.load(client_ckpt_global, weights_only=True))
                print(f"Local and global model loaded successfully from {client_ckpt}")
                    
                self.local_knn_outputs = self.compute_linear_outputs(features, scope="local")
                self.global_knn_outputs = self.compute_linear_outputs(features, scope="global")
            
            elif self.args_.classifier == 'linear_per': 
                self.local_classifier = LinearLayer(self.features_dimension, self.num_classes) 
                self.global_classifier = LinearLayer(self.features_dimension, self.num_classes) 
                
                ckpt_path = os.path.join(self.args_.chkpts_path, 'local_classifier')
                os.makedirs(ckpt_path, exist_ok=True)
                client_ckpt = os.path.join(ckpt_path, f'client_{self.id}.pt')
                client_ckpt_global = os.path.join(self.args_.fedavg_chkpts_dir, f'global_{self.args_.n_fedavg_rounds}.pt')

                if not os.path.isfile(client_ckpt): 
                    self.local_classifier = self.train_linear_classifier(scope="local", model=self.local_classifier, ckpt_path=client_ckpt)

                self.local_classifier.load_state_dict(torch.load(client_ckpt, weights_only=True))
                self.global_classifier.load_state_dict(torch.load(client_ckpt_global, weights_only=True))
                print(f"Local model loaded successfully from {client_ckpt}")
                print(f"Global model loaded successfully from {client_ckpt_global}")
                    
                self.local_knn_outputs = self.compute_linear_outputs(features, scope="local")
                self.global_knn_outputs = self.compute_linear_outputs(features, scope="global")
        
            elif self.args_.classifier == 'knn_per':
                self.global_classifier = LinearLayer(self.features_dimension, self.num_classes) 
                client_ckpt_global = os.path.join(self.args_.fedavg_chkpts_dir, f'global_{self.args_.n_fedavg_rounds}.pt')
                
                self.global_classifier.load_state_dict(torch.load(client_ckpt_global, weights_only=True))
                print(f"Global model loaded successfully from {client_ckpt_global}")
                    
                self.local_knn_outputs = self.compute_knn_outputs(features, scope="local", method=self.knn_weights)
                self.global_knn_outputs = self.compute_linear_outputs(features, scope="global")
        
            if self.local_knn_outputs_flag and self.global_knn_outputs_flag:
                self.local_knn_outputs = softmax(self.local_knn_outputs, axis=1)
                self.global_knn_outputs = softmax(self.global_knn_outputs, axis=1)
                outputs = weight * self.local_knn_outputs + (1 - weight) * self.global_knn_outputs
            elif not self.local_knn_outputs_flag and self.global_knn_outputs_flag:
                warnings.warn("evaluation is done only with global outputs, local datastore is empty", RuntimeWarning)
                outputs = self.global_knn_outputs
            elif self.local_knn_outputs_flag and not self.global_knn_outputs_flag:
                warnings.warn("evaluation is done only with local outputs, global datastore is empty", RuntimeWarning)
                outputs = self.local_knn_outputs
            
        predictions = np.argmax(outputs, axis=1) 
        correct = (labels == predictions).sum()
        total = len(labels)

        acc = correct / total 
        balanced_acc = balanced_accuracy_score(labels, predictions)
        f1_score_macro = f1_score(labels, predictions, average='macro')
        f1_score_weighted = f1_score(labels, predictions, average='weighted')

        num_classes = outputs.shape[1]  # Total number of classes
        unique_classes = np.unique(labels)
        if len(unique_classes) == 1:  # Only one class present
            print("Warning: Only one class present in labels. Returning 0.5 (random performance).")
            roc_auc = 0.5  # Assuming random performance

        elif num_classes > 2:
            # Create a mask to keep only relevant class logits
            mask = np.zeros_like(outputs, dtype=bool)
            mask[:, unique_classes] = True

            # Apply masking before softmax (set irrelevant logits to -inf so they don't affect probabilities)
            masked_outputs = np.where(mask, outputs, -np.inf)

            # Apply softmax AFTER masking, ensuring only relevant classes get proper probabilities
            probabilities = softmax(masked_outputs, axis=1)

            # Compute ROC-AUC only for present classes
            roc_auc = roc_auc_score(labels, probabilities[:, unique_classes], multi_class='ovr', average='micro')

        else: # Binary classification case (assume positive class is at index 1)
            probabilities = softmax(outputs, axis=1)
            roc_auc = roc_auc_score(labels, probabilities[:, 1])

        return acc, balanced_acc, roc_auc, f1_score_macro, f1_score_weighted

    def clear_datastore(self):
        """
        clears local `datastore`
        """
        if self.use_cosine_knn:
            if self.chroma_local_collection:
                self.chroma_client.delete_collection(self.chroma_local_collection.name)
            self.chroma_local_collection = None

        else:
            self.datastore.clear()
            self.datastore.capacity = self.capacity
        
        self.datastore_flag = False
        self.local_knn_outputs_flag = False
        self.global_knn_outputs_flag = False