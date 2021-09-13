from torch.utils.data import Dataset, DataLoader
from FedML.FedSchemes.FedAvg import *
from FedML.Data.datasets import mnist
from FedML.Data.distribute_data import get_iid_mnist, get_non_iid_mnist


mnist_train, mnist_test = mnist()