from collections import Counter
from opacus import GradSampleModule
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import os

from models.cgan import CGANTrainer, Generator, Discriminator
from utils.constants import *

class CGANAnonymizer:
    def __init__(self, args, g):
        """
        :param args: Namespace containing parameters and hyperparameters of the current run.
        """

        self.args = args
        self.g = g
        self.device = self.args.device 
        self.lr = self.args.anonymizer_lr

        self.batch_size = 256
        self.latent_dim = 100
        self.embedding_dim = EMBEDDING_DIM[self.args.backbone] 
        self.num_classes = N_CLASSES[self.args.experiment] 

    def apply(self, client):
        data, labels, val_data, val_labels = client.train_features, client.train_labels, client.val_features, client.val_labels
        train_loader, val_loader = self.prepare_data(data, labels, val_data, val_labels)

        ckpt_path = self.args.chkpts_path 
        if self.args.anonymizer == "cgan_fedavg":
            best_ckpt = os.path.join(ckpt_path, "cgan_fedavg", f'client_{client.id}.pt') 
        else:
            best_ckpt = os.path.join(ckpt_path, f'client_{client.id}.pt') 


        os.makedirs(ckpt_path, exist_ok=True)
        
        generator = Generator(latent_dim=self.latent_dim, label_dim=self.num_classes, embedding_dim=self.embedding_dim).to(self.device)
        discriminator = Discriminator(input_dim=self.embedding_dim, label_dim=self.num_classes).to(self.device)
        g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        trainer = CGANTrainer(generator, discriminator, g_optimizer, d_optimizer, self.args, self.device, best_ckpt, latent_dim=self.latent_dim, lr=self.lr) 
        
        if not os.path.isfile(best_ckpt): 
            trainer.fit(train_loader, val_loader, num_epochs=100, lr_scheduler="multi_step")
            
        trainer.load_checkpoint()

        gen_samples, gen_labels = self.generate_samples(trainer.generator, labels, self.train_min, self.train_max)
        
        return gen_samples, gen_labels

    def prepare_data(self, data, labels, val_data, val_labels):
    
        data, labels = torch.Tensor(data).to(self.device), torch.Tensor(labels).to(self.device).to(int)
        val_data, val_labels = torch.Tensor(val_data).to(self.device), torch.Tensor(val_labels).to(self.device).to(int)

        data, train_min, train_max = self._normalize(data)
        val_data, _, _ = self._normalize(val_data, train_min, train_max)
        self.train_min = train_min
        self.train_max = train_max

        one_hot_train_labels = F.one_hot(labels, num_classes=self.num_classes).to(self.device)
        one_hot_val_labels = F.one_hot(val_labels, num_classes=self.num_classes).to(self.device)

        train_dataset = TensorDataset(data, one_hot_train_labels)
        val_dataset = TensorDataset(val_data, one_hot_val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True, generator=self.g) #Original: batch_size = 512
        val_loader = DataLoader(val_dataset, batch_size = self.batch_size, shuffle = False, generator=self.g) #Original: batch_size = 512

        return train_loader, val_loader


    def _get_replica_class_distribution(self, labels, factor=1.0):
        """
        Return a tensor indicating how many samples to generate per class by 
        multiplying the original class counts by `factor`.
        """
        label_counts = Counter(labels)
        samples_per_class = torch.zeros(self.num_classes, dtype=torch.long)

        for label, count in label_counts.items():
            samples_per_class[label] = int(count * factor)

        return samples_per_class

    def _get_samples_per_class(self, labels, total_generated, strategy):
        label_counts = Counter(labels)

        # Count per class
        spc = torch.tensor([label_counts.get(i, 0) for i in range(self.num_classes)], dtype=torch.float)
        mask = spc > 0 # Classes present in the data

        # Compute inverse-frequency weights only for present classes
        weights = torch.zeros_like(spc)

        if strategy == "inverse":
            weights[mask] = 1.0 / spc[mask]  # More samples for smaller classes
        elif strategy == "uniform":
            weights[mask] = 1.0  # Equal weight for all present classes
        elif strategy == "uniform_all_classes":
            weights[:] = 1.0  # Equal weight for all classes, incl. classes missing in train data
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Normalize and compute number of samples per class
        probs = weights / weights.sum()
        samples_per_class = (probs * total_generated).long()
        return samples_per_class

    def generate_samples(self, generator, real_labels, train_min, train_max, var=1.0, total_generated_factor=1.0, strategy='inverse'):
        gen_samples, gen_labels = [], []

        total_real_samples = len(real_labels)
        if strategy == "replicate":
            samples_per_class = self._get_replica_class_distribution(real_labels, total_generated_factor)
            total_generated_samples = samples_per_class.sum().item()
        else:
            total_generated_samples = total_generated_factor * total_real_samples  
            samples_per_class = self._get_samples_per_class(real_labels, total_generated_samples, strategy)

        generator.eval()  
        generator.to(self.device) 

        with torch.no_grad():
            # Generate all latent samples at once
            all_labels = []
            all_latents = []

            for label in range(self.num_classes):
                n_samples = samples_per_class[label].item()
                if n_samples == 0:
                    continue

                all_labels.append(torch.full((n_samples,), label, dtype=torch.long, device=self.device))
                all_latents.append(var * torch.randn((n_samples, self.latent_dim), device=self.device))

            all_labels = torch.cat(all_labels)  # Shape: (total_samples,)
            all_latents = torch.cat(all_latents)  # Shape: (total_samples, latent_dim)
            
            one_hot_labels = F.one_hot(all_labels, num_classes=self.num_classes).to(self.device)

            x_hat = generator(all_latents, one_hot_labels).to(self.device)
            x_hat = self._un_normalize(x_hat, train_min, train_max)

            gen_samples = x_hat.cpu().numpy()
            gen_labels = all_labels.cpu().numpy().tolist()

        return gen_samples, gen_labels


    def _normalize(self, tensor, _min = None, _max = None):
        """
        Normalize tensor.

        :param tensor: embedding to normalize.
        :param _min: minimum value for normalization.
        :param _max: minimum value for normalization.
        :return normalized tensor.
        """
        if (_min == None) and (_max == None):
            _min = tensor.min(dim=0).values
            _max = tensor.max(dim=0).values
        return (tensor - _min) / (_max - _min), _min, _max


    def _un_normalize(self, tensor, _min, _max):
        """
        Un-normalize tensor.

        :param tensor: embedding to normalize.
        :param _min: minimum value for un-normalizing.
        :param _max: minimum value for un-normalizing.
        :return un-normalized tensor.
        """
        return (tensor) * (_max - _min) + _min
    
    # FedAvg on CGAN:
    def get_trainer(self, client, global_model=None, is_trained=True, num_epochs=None):
        ckpt_path = self.args.chkpts_path 
        if client:
            best_ckpt = os.path.join(ckpt_path, "cgan_fedavg", f"client_{client.id}.pt") 
        else: 
            best_ckpt = os.path.join(ckpt_path, "cgan_fedavg", f"global.pt") 

        os.makedirs(ckpt_path, exist_ok=True)
        
        
        if not is_trained:
            generator = Generator(latent_dim=self.latent_dim, label_dim=self.num_classes, embedding_dim=self.embedding_dim).to(self.device)
            discriminator = Discriminator(input_dim=self.embedding_dim, label_dim=self.num_classes).to(self.device)
            g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
            d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

            trainer = CGANTrainer(
                generator=generator, 
                discriminator=discriminator, 
                g_optimizer=g_optimizer, 
                d_optimizer=d_optimizer, 
                args=self.args, 
                device=self.device, 
                ckpt_path=best_ckpt, 
                latent_dim=self.latent_dim,
                num_epochs=num_epochs,
                lr=self.lr
                ) 
            

            return trainer

        train_loader, val_loader = self.prepare_data(client.train_features, client.train_labels, client.val_features, client.val_labels)

        client.trainer.fit(train_loader, val_loader, global_model=global_model, num_epochs=num_epochs, lr_scheduler=None) 

        return client.trainer
    
