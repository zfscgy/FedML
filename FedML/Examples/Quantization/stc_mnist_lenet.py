"""
 This code mainly uses the method described in TernGrad
"""
from typing import List

import torch
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset
from FedML.Models import LeNet5
from FedML.Base.Utils import get_tensors
from FedML.FedSchemes.fedavg import *
from FedML.FedSchemes.Quantization.sparse_ternary_compression import STC, STCClientOptions, STCClient, \
    STCServerOptions, STCServer
from FedML.Data.datasets import Mnist
from FedML.Data.distribute_data import get_iid_mnist
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


stc_100 = STC(0.01)


def stc_ternarize(model: nn.Module):
    tensor_list = get_tensors(model)
    return stc_100.quantize_tensor_list(tensor_list)


server = STCServer(
    lambda: LeNet5(),
    STCServerOptions(
        n_clients_per_round=10,
        quantize=stc_ternarize
    )
)


cross_entropy = nn.CrossEntropyLoss()


def loss_func(ys_pred, ys_true):
    return cross_entropy(ys_pred, torch.argmax(ys_true, dim=-1))


clients = []
for i in range(n_clients):
    client = STCClient(
        lambda: LeNet5(),
        server,
        STCClientOptions(
            client_data_loader=DataLoader(
                TensorDataset(torch.from_numpy(iid_mnist_datasets[i][:, :784]).float().view(-1, 1, 28, 28),
                              torch.from_numpy(iid_mnist_datasets[i][:, 784:])), batch_size=20),
            get_optimizer=lambda m: SGD(m.parameters(), 0.04),
            loss_func=loss_func,
            batch_mode=False,
            n_local_rounds=1,
            ternarize=stc_ternarize
        )
    )
    clients.append(client)


server.set_clients(clients)


def multiclass_acc(xs, ys):
    return np.mean(np.argmax(xs, axis=-1) == np.argmax(ys, axis=-1))


fed_train = FedTrain(server)

fed_train.start()

fed_train.train(
    n_global_rounds=1000,
    test_per_global_rounds=10,
    test_data_loader=DataLoader(mnist_test, batch_size=128),
    test_metrics=[multiclass_acc]
)
