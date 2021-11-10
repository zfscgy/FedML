"""
 This code mainly uses the method described in TernGrad
"""


from typing import List

import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset
from FedML.Models import LeNet5
from FedML.FedSchemes.fedavg import *
from FedML.FedSchemes.Quantization.ternary_weight_network import TernaryServerOptions, TernaryServer_GradientAvg
from FedML.FedSchemes.Quantization.terngrad import TernGrad
from FedML.Data.datasets import Mnist
from FedML.Data.distribute_data import get_iid_mnist
from FedML.Train import FedTrain
from FedML.Base.Utils import get_tensors_by_function


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


def terngrad_ternarize(x: List[torch.Tensor]):
    ternarized, compression_rate = TernGrad().quantize_tensor_list(x)
    return [t * 0.01 for t in ternarized], compression_rate


server = TernaryServer_GradientAvg(
    lambda: LeNet5(),
    TernaryServerOptions(
        n_clients_per_round=10,
        ternarize_server=lambda x: (x, 1.),
        ternarize_client=terngrad_ternarize
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
