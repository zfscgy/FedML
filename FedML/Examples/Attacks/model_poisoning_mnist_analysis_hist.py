import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from FedML.Models import LeNet5
from FedML.Data.datasets import Mnist
from FedML.FedSchemes.Attacks.model_poisoning import FedAvgServer, FedAvgClient, MaliciousClient, \
    FedAvgServerOptions, FedAvgClientOptions, MaliciousClientOptions
from FedML.Data.distribute_data import get_iid_mnist
from FedML.Train import FedTrain
from FedML.Base.Utils import test_on_data_loader, Convert, get_tensors_by_function

import matplotlib.pyplot as plt


def plot_gradient_distribution(grads: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(grads.flatten(), bins=200)
    return fig


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

"""
 Let the last dataset shard to be 'malicious dataset'
 (Note: Actually only 10 samples are used, while other samples are dropped)
"""
malicious_dataset = iid_mnist_datasets.pop()
# Print first 10 true labels of the malicious dataset
print(f"Original labels of malicious data: {np.argmax(malicious_dataset[:10, 784:], axis=-1)}")
malicious_dataset[:, 784:] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # change all samples to label 0

server = FedAvgServer(
    lambda: LeNet5(),
    FedAvgServerOptions(
        n_clients_per_round=10
    )
)

cross_entropy = nn.CrossEntropyLoss()


def loss_func(ys_pred, ys_true):
    return cross_entropy(ys_pred, torch.argmax(ys_true, dim=-1))


benign_clients = []
for i in range(n_clients - 1):
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
    benign_clients.append(client)




malicious_client = MaliciousClient(
    lambda: LeNet5(),
    server,
    MaliciousClientOptions(
        client_data_loader=DataLoader(
            TensorDataset(torch.from_numpy(iid_mnist_datasets[-1][:, :784]).float().view(-1, 1, 28, 28),
                          torch.from_numpy(iid_mnist_datasets[-1][:, 784:])), batch_size=50),
        get_optimizer=lambda m: Adam(m.parameters()),
        loss_func=loss_func,
        batch_mode=False,
        n_local_rounds=1,
        n_local_rounds_malicious=5,
        malicious_data_loader=DataLoader(  # Only use 10 samples as malicious samples (Otherwise may not work).
            TensorDataset(torch.from_numpy(malicious_dataset[:10, :784]).float().view(-1, 1, 28, 28),
                          torch.from_numpy(malicious_dataset[:10, 784:])), batch_size=128),
        malicious_boost=10
    ),
)


server.set_clients(benign_clients + [malicious_client])


def multiclass_acc(xs, ys):
    return np.mean(np.argmax(xs, axis=-1) == np.argmax(ys, axis=-1))


def plot_update_hists():
    if malicious_client not in server.current_training_clients:
        return

    print(f"=========Malicious client is chosen=============")

    normal_hists = []
    for client in server.current_training_clients:
       if client is malicious_client:
           malicious_hist = plot_client_update_hist(client)
       else:
           normal_hists.append(plot_client_update_hist(client))

    malicious_hist.suptitle("Malicious Update Distribution")
    malicious_hist.show()
    for hist in normal_hists:
        hist.show()


def calculate_malicious_metrics():
    acc = test_on_data_loader(server.global_model, malicious_client.options.malicious_data_loader, [multiclass_acc])[0]
    confidence = Convert.to_numpy(F.softmax(Convert.model_to_device(server.global_model)(
        Convert.to_tensor(malicious_dataset[:10, :784]).float().view(-1, 1, 28, 28)), dim=-1))
    Convert.model_to_cpu(server.global_model)
    print(f"Malicious accuracy: {acc:.3f}")
    print(f"Malicious confidence: {confidence[:, 0]}")


def plot_client_update_hist(client):
    update = Convert.to_numpy(list(server.global_model.parameters())[0] - list(client.local_model.parameters())[0])
    return plot_gradient_distribution(update)



fed_train = FedTrain(server)


fed_train.train(
    n_global_rounds=1000,
    test_per_global_rounds=1,
    test_data_loader=DataLoader(mnist_test, batch_size=128),
    test_metrics=[multiclass_acc],
    round_callback=plot_update_hists,
    test_callback=calculate_malicious_metrics
)