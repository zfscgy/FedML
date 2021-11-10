"""
 This code mainly uses the method described in TernGrad
"""
from typing import List

import torch
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt

from FedML.Models.simple_models import MnistLogistic
from FedML.Base.Utils import get_tensors
from FedML.FedSchemes.fedavg import *
from FedML.FedSchemes.Quantization.sparse_ternary_compression import STC, STCClientOptions, STCClient, \
    STCServerOptions, STCServer
from FedML.FedSchemes.Experimental.learned_stc import LearnedSTCClientOptions, LearnedSTCClient
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
mnist_train, mnist_test = Mnist.get(txs=[Mnist.tx_flatten], tys=[Mnist.ty_onehot])



"""
 Train in iid setting
 Notice: The generated mnist train data must also be in range (0, 1) in order to be consistent with the test dataset!
"""
iid_mnist_datasets = get_iid_mnist(np.concatenate([mnist_train.data.view(-1, 784).float().numpy() / 255,
                                                   torch.eye(10)[mnist_train.targets].numpy()], axis=-1),
                                   samples_per_client)


stc_100 = STC(1/400)


def stc_ternarize(model: nn.Module):
    tensor_list = get_tensors(model)
    return stc_100.quantize_tensor_list(tensor_list)



cross_entropy = nn.CrossEntropyLoss()


def loss_func(ys_pred, ys_true):
    return cross_entropy(ys_pred, torch.argmax(ys_true, dim=-1))


def multiclass_acc(xs, ys):
    return np.mean(np.argmax(xs, axis=-1) == np.argmax(ys, axis=-1))


def stc_test():
    server = STCServer(
        lambda: MnistLogistic(),
        STCServerOptions(
            n_clients_per_round=10,
            quantize=stc_ternarize
        )
    )

    clients = []
    for i in range(n_clients):
        client = STCClient(
            lambda: MnistLogistic(),
            server,
            STCClientOptions(
                client_data_loader=DataLoader(
                    TensorDataset(torch.from_numpy(iid_mnist_datasets[i][:, :784]).float(),
                                  torch.from_numpy(iid_mnist_datasets[i][:, 784:])), batch_size=100),
                get_optimizer=lambda m: SGD(m.parameters(), 0.04),
                loss_func=loss_func,
                batch_mode=True,
                n_local_rounds=32,
                quantize=stc_ternarize
            )
        )
        clients.append(client)
    server.set_clients(clients)


    fed_train = FedTrain(server)

    fed_train.start()

    test_records = fed_train.train(
        n_global_rounds=100,
        test_per_global_rounds=1,
        test_data_loader=DataLoader(mnist_test, batch_size=128),
        test_metrics=[multiclass_acc]
    )
    return test_records


def learned_stc_test():
    server = STCServer(
        lambda: MnistLogistic(),
        STCServerOptions(
            n_clients_per_round=10,
            quantize=stc_ternarize
        )
    )

    clients = []
    for i in range(n_clients):
        client = LearnedSTCClient(
            lambda: MnistLogistic(),
            server,
            LearnedSTCClientOptions(
                client_data_loader=DataLoader(
                    TensorDataset(torch.from_numpy(iid_mnist_datasets[i][:, :784]).float(),
                                  torch.from_numpy(iid_mnist_datasets[i][:, 784:])), batch_size=100),
                get_optimizer=lambda m: SGD(m.parameters(), 0.04),
                loss_func=loss_func,
                batch_mode=True,
                n_local_rounds=32,
                quantize=stc_ternarize,
                lambda_l1=40
            )
        )
        clients.append(client)
    server.set_clients(clients)

    fed_train = FedTrain(server)

    fed_train.start()

    test_records = fed_train.train(
        n_global_rounds=100,
        test_per_global_rounds=1,
        test_data_loader=DataLoader(mnist_test, batch_size=128),
        test_metrics=[multiclass_acc]
    )
    return test_records


print(">>>>>STC")
records_stc = np.array(stc_test())

print(">>>>>Learned STC")
records_learned_stc = np.array(learned_stc_test())


plt.plot(records_stc[:, 0], records_stc[:, 1], "o")
plt.plot(records_learned_stc[:, 0], records_learned_stc[:, 1], '.')
plt.show()