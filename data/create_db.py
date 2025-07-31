"""
Adapted from: https://github.com/francescodisalvo05/cvae-anonymization/blob/main/create_db.py 
"""

from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from wilds import get_dataset

from argparse import ArgumentParser
from tqdm import tqdm

import torchvision.transforms as T
import torchvision
import torch

import numpy as np
import timm
import os

import sys
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
# adding the root directory to the sys.path.
sys.path.append(root)

from utils.utils import *

import medmnist
from medmnist import INFO


class CustomCamelyon(Dataset):
    def __init__(self, root, transform):
        self.dataset = get_dataset(dataset="camelyon17", root_dir=root, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label, metadata = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img) 
        return img, label, metadata
    
def get_dataloaders_by_hospitals(args, dataset):
    """
    Get the dataloaders for the selected dataset and backbone.
    :param args: Namespace containing parameters and hyperparameters of the current run.
    :param dataset: Downloaded Camelyon17 dataset
    :return dataloaders: Dictionary containing a dataloader for each hospital's data split.
    """
    # UNCOMMENT IF CREATING FROM SCRATCH!!!
    # # Dictionary to store indices for each hospital
    # hospital_indices = {f"client_{i}": [] for i in range(5)}  

    # # Collect sample indices per hospital
    # for idx in range(len(dataset)):
    #     _, _, metadata = dataset.__getitem__(idx)
    #     hospital_id = f"client_{int(metadata[0])}" #int(metadata[0])  
    #     hospital_indices[hospital_id].append(idx)
    
    # with open(os.path.join("data/camelyon17/database/", "hospital_indices.pkl"), "wb") as f:
    #     pickle.dump(hospital_indices, f)
    #     #Debug
    #     print("Indices saved successfully")

    with open(os.path.join("data/camelyon17/database/", "hospital_indices.pkl"), "rb") as f:
        hospital_indices = pickle.load(f)

    # Create DataLoaders for each hospital
    dataloaders = {}

    for hospital_id, indices in hospital_indices.items():
        # Create a Subset for each hospital's data
        subset = Subset(dataset, indices)
        # Create a DataLoader for the subset
        dataloaders[hospital_id] = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    #Debug
    # Print dataloader keys to verify
    print("Dataloader names:", dataloaders.keys())
    # Example access to a specific hospital's dataloader
    loader_client_0 = dataloaders["client_0"]
    print(f"Number of batches for Hospital 0: {len(loader_client_0)}")
    # Test one batch to verify
    for images, labels, metadata in loader_client_0:
        print(f"Sample batch from Hospital 0 - Images shape: {images.shape}, Labels shape: {labels.shape}, Metadata: {metadata[0]}")
        break

    return dataloaders


def get_dataloaders(args, transforms):
    """
    Get the dataloaders for the selected dataset and backbone.
    :param args: Namespace containing parameters and hyperparameters of the current run.
    :param transforms: Transform function. 
    :return dataloaders: Dictionary containing both train and test dataloaders.
    """

    data_path = args.dataset_root

    if args.dataset == "camelyon17":
        dataset = CustomCamelyon(root=data_path, transform=transforms)
        return get_dataloaders_by_hospitals(args=args, dataset=dataset)

    # convert img to RGB & append after ToTensor() 
    #   this is used for MNIST and F-MNIST
    to_rgb = T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x) 

    if args.dataset == "mnist":
        transforms.transforms.insert(-1, to_rgb)
        trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms)
        testset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms)
    elif args.dataset in ["organamnist", "dermamnist", "organcmnist", "retinamnist", "organsmnist", "pathmnist", "pneumoniamnist", "bloodmnist"]:
        info = INFO[args.dataset] 
        DataClass = getattr(medmnist, info['python_class']) # get relative dataset class
        trainset = DataClass(split='train', transform=transforms, download=False, as_rgb=True, size=224, root=data_path, mmap_mode='r')
        valset = DataClass(split='val', transform=transforms, download=False, as_rgb=True, size=224, root=data_path, mmap_mode='r')
        testset = DataClass(split='test', transform=transforms, download=False, as_rgb=True, size=224, root=data_path, mmap_mode='r')

        # Concatenate train and val sets into a single training set
        trainset = ConcatDataset([trainset, valset])

    else:
        raise ValueError(f"{args.dataset} not available")
    
    dataloaders = {}

    trainloader = DataLoader(trainset, batch_size = args.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    testloader = DataLoader(testset, batch_size = args.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    dataloaders['train'] = trainloader
    dataloaders['test'] = testloader

    return dataloaders

def extract_embeddings(model, device, dataset, dataloader):
    """
    Make inference on the given dataset through the chosen backbone.
    :param model: Selected backbone.
    :param device: Running device.
    :param dataset: Name of the dataset.
    :param dataloader: Current dataloader (train|test).
    :return data: Dictionary containing the extracted data.
    """

    embeddings_db, labels_db, metadata_db, domains_db = [], [], [], []

    for extracted in tqdm(dataloader):
        if dataset == "camelyon17":
            images, labels, metadata = extracted
        else:
            images, labels = extracted
        
        images = images.to(device)

        output = model.forward_features(images)
        output = model.forward_head(output, pre_logits=True)

        labels_db.extend(labels)
        embeddings_db.extend(output.detach().cpu().numpy())
        if dataset == "camelyon17":
            metadata_db.extend(metadata)
    
    data = {
        'embeddings': embeddings_db,
        'labels': labels_db
    }

    if dataset == "camelyon17":
        data['metadata'] = metadata_db

    return data



def main(args):
    seed_everything(args.seed)
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # get model from timm
    model = timm.create_model(args.backbone, pretrained=True, num_classes=0).to(device)
    model.requires_grad_(False)
    model = model.eval()

    # get the required transform function for the given feature extractor
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # get dataloaders
    dataloaders = get_dataloaders(args, transforms)

    # create database folder, if necessary
    os.makedirs(os.path.join(args.database_root),exist_ok=True)

    if args.dataset == "camelyon17":
        splits = ['client_0','client_1','client_2','client_3','client_4']
    else:
        splits = ['train','test']

    #for split in ['train','test']:
    for split in splits:
        if args.dataset == "epistroma":
            for data_split in ['train','test']:
                db = extract_embeddings(model = model, 
                                        device = device,
                                        dataset = args.dataset,
                                        dataloader = dataloaders[split][data_split])
                
                # store database: database_root / client_[id] / train|test.npz
                os.makedirs(os.path.join(args.database_root, split), exist_ok=True)
                np.savez(os.path.join(args.database_root, split, f'{data_split}.npz'), **db)

        else:
            # get database of embeddings in the form
            #   db = {'embeddings' : [...], 'labels' : [...], ...}
            db = extract_embeddings(model = model, 
                                    device = device,
                                    dataset = args.dataset,
                                    dataloader = dataloaders[split])
        
            # store database: database_root / train|test.npz
            np.savez(os.path.join(args.database_root,f'{split}.npz'), **db)


if __name__ == '__main__':

    parser = ArgumentParser()

    # GENERAL
    parser.add_argument('--dataset_root', type=str, default="tmp/assets/dataset", help='define the dataset root') #e.g. dataset_root="data/organsmnist/dataset/"
    parser.add_argument('--database_root', type=str, default="tmp/assets/database", help='define the database root') #e.g. database_root="data/organsmnist/database/"

    # DATASET & HYPERPARAMS
    parser.add_argument('--dataset', type=str, required=True, help='define the dataset name') #e.g. dataset="cifar10"
    parser.add_argument('--backbone', type=str, default='vit_base_patch14_dinov2.lvd142m', help='define the feature extractor') #vit_small_patch16_224.dino vit_small_patch14_dinov2.lvd142m vit_small_patch16_224.augreg_in21k_ft_in1k 
    parser.add_argument('--batch_size', type=int, default=128, help='define the batch size')
    parser.add_argument('--seed', type=int, default=42, help='define the random seed')

    # ADD METHOD
    args = parser.parse_args()

    main(args)