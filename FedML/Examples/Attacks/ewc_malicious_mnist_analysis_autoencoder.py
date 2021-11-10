import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

from FedML.Models import LeNet5, AutoEncoder
from FedML.Data.datasets import Mnist
from FedML.FedSchemes.Experimental.malicious_via_autoencoder import FedAvgServer, FedAvgClient, EWCMaliciousClient, \
    FedAvgServerOptions, FedAvgClientOptions, EWCMaliciousClientOptions
from FedML.Data.distribute_data import get_iid_mnist
from FedML.Train import FedTrain
from FedML.Base.Utils import test_on_data_loader, Convert, get_tensors_by_function, train_n_epochs



"""
 Some settings
"""
samples_per_client = 3000
n_clients = 20


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


round_nums = []
malicious_stds = []
mean_stds = []
max_stds = []
malicious_errors = []
mean_errors = []
max_errors = []

def test_autoencoder():
    if malicious_client not in server.current_training_clients:
        return
    malicious_index = server.current_training_clients.tolist().index(malicious_client)
    losses = [[] for _ in auto_encoders]
    all_updates = [get_tensors_by_function([client.local_model.parameters(), server.global_model.parameters()],
                                           lambda xs: xs[0] - xs[1])
               for client in server.current_training_clients]

    print(f"Current malicious client index: {malicious_index}")

    round_nums.append(server.current_global_rounds)
    malicious_stds.append([])
    mean_stds.append([])
    max_stds.append([])
    malicious_errors.append([])
    mean_errors.append([])
    max_errors.append([])

    for i, auto_encoder in enumerate(auto_encoders):
        param_updates = torch.stack([update[i].flatten() for update in all_updates])
        stds = np.std(Convert.to_numpy(param_updates), axis=-1)

        mean_std = np.mean(stds)
        mal_std = stds[malicious_index]
        stds[malicious_index] = -999
        max_std = np.max(stds)

        print(f"Std mean: {mean_std}, max: {max_std}, mal: {mal_std}")
        decoder_outputs = Convert.model_to_device(auto_encoder)(Convert.to_tensor(param_updates))
        mses = torch.mean(torch.square(decoder_outputs - Convert.to_tensor(param_updates)), dim=-1)
        mses = Convert.to_numpy(mses)

        mean_error = np.mean(mses)
        mal_error = mses[malicious_index]
        mses[malicious_index] = -999
        max_error = np.max(mses)

        print(f"Current Parameter Reconstruction Losses: {mses.tolist()}\n "
              f"mean loss: {mean_error}, max Loss: {max_error} mal: {mal_error}")
        losses.append(mses)

        malicious_stds[-1].append(mal_std)
        mean_stds[-1].append(mean_std)
        max_stds[-1].append(max_std)
        malicious_errors[-1].append(mal_error)
        mean_errors[-1].append(mean_error)
        max_errors[-1].append(max_error)


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
            n_local_rounds=1
        )
    )
    benign_clients.append(client)




malicious_client = EWCMaliciousClient(
    lambda: LeNet5(),
    server,
    EWCMaliciousClientOptions(
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
        ewc_reg_term=5
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
    n_global_rounds=100,
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
    test_per_global_rounds=10,
    test_data_loader=DataLoader(mnist_test, batch_size=128),
    test_metrics=[multiclass_acc],
    round_callback=test_autoencoder,
    test_callback=calculate_malicious_metrics
)


max_stds = np.array(max_stds)
mean_stds = np.array(mean_stds)
malicious_stds = np.array(malicious_stds)
max_errors = np.array(max_errors)
mean_errors = np.array(mean_errors)
malicious_errors = np.array(malicious_errors)


for i in range(8):
    plt.plot(round_nums, max_stds[:, i])
    plt.plot(round_nums, mean_stds[:, i])
    plt.plot(round_nums, malicious_stds[:, i], marker="x")
    plt.title(f"STD of parameter group {i}")
    plt.xlabel("No. Rounds")
    plt.show()

    plt.plot(round_nums, max_errors[:, i])
    plt.plot(round_nums, mean_errors[:, i])
    plt.plot(round_nums, malicious_errors[:, i], marker="x")
    plt.title(f"Error of parameter group {i}")
    plt.xlabel("No. Rounds")
    plt.show()
