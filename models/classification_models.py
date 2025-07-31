import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class LinearLayer(nn.Module):
    def __init__(self, input_dimension, num_classes, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dimension, num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dimension, num_classes, hidden_dims=[512, 256], bias=True, dropout=0.3):
        super(MultiLayerPerceptron, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes

        layers = []
        prev_dim = input_dimension

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=bias))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes, bias=bias))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def get_mobilenet(n_classes, pretrained=True):
    """
    creates MobileNet model with `n_classes` outputs

    :param n_classes:
    :param pretrained: (bool)

    :return:
        model (nn.Module)

    """
    model = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)

    return model
