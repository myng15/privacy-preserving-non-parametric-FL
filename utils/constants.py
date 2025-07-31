import string


ALL_STRATEGIES = {
    "random"
}

ALL_MODELS = {
    "mobilenet",
    "linear",
    "mlp"
}

LOADER_TYPE = {
    "mnist": "mnist",
    "organamnist": "medmnist", 
    "organcmnist": "medmnist", 
    "organsmnist": "medmnist", 
    "dermamnist": "medmnist", 
    "retinamnist": "medmnist", 
    "pathmnist": "medmnist",
    "bloodmnist": "medmnist",
    "pneumoniamnist": "medmnist",
    "camelyon17": "camelyon17",
}

EXTENSIONS = {
    "mnist": ".pkl",
    "medmnist": ".pkl",
}

N_CLASSES = {
    "mnist": 10,
    "organamnist": 11,
    "organcmnist": 11,
    "organsmnist": 11,
    "dermamnist": 7,
    "retinamnist": 5,
    "pathmnist": 9,
    "bloodmnist": 8,
    "pneumoniamnist": 2,
    "camelyon17": 2,
}

EMBEDDING_DIM = {
    "base_patch14_dinov2": 768, #"vit_base_patch14_dinov2.lvd142m"
    "base_patch16_dino": 768, #"vit_base_patch16_224.dino"
    "base_patch16_augreg": 768, #"vit_base_patch16_224.augreg_in21k_ft_in1k"
    "small_patch14_dinov2": 384, #"vit_small_patch14_dinov2.lvd142m"
    "small_patch16_dino": 384, #"vit_small_patch16_224.dino"
    "small_patch16_augreg": 384, #"vit_small_patch16_224.augreg_in21k_ft_in1k"
}

LOCAL_HEAD_UPDATES = 10  # number of epochs for local heads used in FedRep

# NUM_WORKERS = os.cpu_count()  # number of workers used to load data and in GPClassifier
NUM_WORKERS = 1
