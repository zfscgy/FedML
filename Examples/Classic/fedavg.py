import numpy as np
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, TensorDataset, DataLoader
from FedML.Models import LeNet5
from FedML.FedSchemes.fedavg import *
from FedML.Data.datasets import Mnist
from FedML.Data.distribute_data import get_iid_mnist, get_non_iid_mnist
from FedML.Train import FedTrain


"""
 Some settings
"""
samples_per_client = 600
n_clients = 100


"""
 Get datasets
"""
mnist_train, mnist_test = Mnist.get(tys=[Mnist.ty_onehot])



"""
 Train in iid setting
 Notice: The generated mnist train data must also be in range (0, 1) in order to be consistent with the test dataset!
"""
iid_mnist_datasets = get_iid_mnist(np.concatenate([mnist_train.data.view(-1, 784).float().numpy() / 255,
                                                   torch.eye(10)[mnist_train.targets].numpy()], axis=-1),
                                   samples_per_client)


server = FedAvgServer(
    lambda: LeNet5(),
    FedAvgServerOptions(
        n_clients_per_round=10,
    )
)

cross_entropy = nn.CrossEntropyLoss()


def loss_func(ys_pred, ys_true):
    return cross_entropy(ys_pred, torch.argmax(ys_true, dim=-1))


clients = []
for i in range(n_clients):
    client = FedAvgClient(
        lambda: LeNet5(),
        server,
        FedAvgClientOptions(
            client_data_loader=DataLoader(
                TensorDataset(torch.from_numpy(iid_mnist_datasets[i][:, :784]).float().view(-1, 1, 28, 28),
                              torch.from_numpy(iid_mnist_datasets[i][:, 784:])), batch_size=50),
            get_optimizer=lambda m: Adam(m.parameters()),
            loss_func=loss_func,
            batch_mode=False,
            n_local_rounds=5
        )
    )
    clients.append(client)

server.set_clients(clients)


def multiclass_acc(xs, ys):
    return np.mean(np.argmax(xs, axis=-1) == np.argmax(ys, axis=-1))


fed_train = FedTrain(server)


fed_train.train(
    n_global_rounds=1000,
    test_per_global_rounds=1,
    test_data_loader=DataLoader(mnist_test, batch_size=128),
    test_metrics=[multiclass_acc]
)
