import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from FedML.Models import LeNet5, AutoEncoder
from FedML.Data.datasets import Mnist
from FedML.FedSchemes.Attacks.model_poisoning import FedAvgServer, FedAvgClient, MaliciousClient, \
    FedAvgServerOptions, FedAvgClientOptions, MaliciousClientOptions
from FedML.Data.distribute_data import get_iid_mnist
from FedML.Train import FedTrain
from FedML.Base.Utils import test_on_data_loader, Convert, get_tensors_by_function, train_n_epochs

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


auto_encoders = [AutoEncoder(p.numel(), 8) for p in server.global_model.parameters()]
all_updates = [[] for _ in server.global_model.parameters()]


def store_updates():
    for client in server.current_training_clients:
        update_params = get_tensors_by_function([client.local_model.parameters(), server.global_model.parameters()],
                                                lambda xs: xs[0] - xs[1])
        for i, p in enumerate(update_params):
            all_updates[i].append(p)


def train_autoencoder():
    for i in range(len(auto_encoders)):
        print(f"Train autoencoder for param {i}...")
        params = torch.stack([p.flatten() for p in all_updates[i]])
        dataset = TensorDataset(params, params)
        train_n_epochs(auto_encoders[i], Adam(auto_encoders[i].parameters()), nn.MSELoss(), DataLoader(dataset), 5)


def test_autoencoder():
    if malicious_client not in server.current_training_clients:
        return
    malicious_index = server.current_training_clients.tolist().index(malicious_client)
    losses = [[] for _ in auto_encoders]
    all_updates = [get_tensors_by_function([client.local_model.parameters(), server.global_model.parameters()],
                                           lambda xs: xs[0] - xs[1])
               for client in server.current_training_clients]

    print(f"Current malicious client index: {malicious_index}")
    for i, auto_encoder in enumerate(auto_encoders):
        param_updates = torch.stack([update[i].flatten() for update in all_updates])
        print(np.std(Convert.to_numpy(param_updates), axis=-1))
        decoder_outputs = Convert.model_to_device(auto_encoder)(Convert.to_tensor(param_updates))
        mses = torch.mean(torch.square(decoder_outputs - Convert.to_tensor(param_updates)), dim=-1)
        mses = Convert.to_numpy(mses)
        print(f"Current Parameter Reconstruction Losses: {mses.tolist()}, "
              f"mean loss: {np.mean(mses)}, "
              f"max Loss: {np.max(mses)} index: {np.argmax(mses)}")
        losses.append(mses)



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
        malicious_boost=0.5
    ),
)


server.set_clients(benign_clients + [malicious_client])


def multiclass_acc(xs, ys):
    return np.mean(np.argmax(xs, axis=-1) == np.argmax(ys, axis=-1))


def calculate_malicious_metrics():
    acc = test_on_data_loader(server.global_model, malicious_client.options.malicious_data_loader, [multiclass_acc])[0]
    confidence = Convert.to_numpy(F.softmax(Convert.model_to_device(server.global_model)(
        Convert.to_tensor(malicious_dataset[:10, :784]).float().view(-1, 1, 28, 28)), dim=-1))
    Convert.model_to_cpu(server.global_model)
    print(f"Malicious accuracy: {acc:.3f}")
    print(f"Malicious confidence: {confidence[:, 0]}")


fed_train = FedTrain(server)


fed_train.train(
    n_global_rounds=30,
    test_per_global_rounds=1,
    test_data_loader=DataLoader(mnist_test, batch_size=128),
    test_metrics=[multiclass_acc],
    round_callback=store_updates,
    test_callback=calculate_malicious_metrics
)

print("First stage global training ended, start training autoencoder")

train_autoencoder()

print("Autoencoders training ended, start second stage")

fed_train.train(
    n_global_rounds=100,
    test_per_global_rounds=1,
    test_data_loader=DataLoader(mnist_test, batch_size=128),
    test_metrics=[multiclass_acc],
    round_callback=test_autoencoder,
    test_callback=calculate_malicious_metrics
)