"""
 This code mainly uses the method described in TernGrad
"""

import pandas as pd
import torch
from torch.optim import Adam

from FedML.Base.Utils import Convert
from FedML.Base.Utils.multi_process_run import parallel_process
from FedML.Data.datasets import Mnist
from FedML.Data.distribute_data import SubDataset, random_distribute
from FedML.FedSchemes.Experimental.learned_stc import LearnedSTCClientOptions, LearnedSTCClient, \
    LearnedSTCServerOptions, LearnedSTCServer
from FedML.FedSchemes.Compression.sparse_ternary_compression import STC, BaseCompressionClientOptions, STCClient, \
    BaseCompressionServerOptions, STCServer
from FedML.FedSchemes.fedavg import *
from FedML.Models.simple_models import SimpleDNN
from FedML.Train import FedTrain

print(torch.version.cuda)

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
iid_mnist_datasets = random_distribute(mnist_train, samples_per_client)
mnist_val = SubDataset(mnist_test, np.arange(1000).tolist())
mnist_test = SubDataset(mnist_test, np.arange(1000, 10000).tolist())

cross_entropy = nn.CrossEntropyLoss()


def loss_func(ys_pred, ys_true):
    return cross_entropy(ys_pred, torch.argmax(ys_true, dim=-1))


def multiclass_acc(xs, ys):
    return np.mean(np.argmax(xs, axis=-1) == np.argmax(ys, axis=-1))


random_seed = np.random.randint(0, 2 ** 20)


n_global_rounds = 2000
n_local_rounds = 128

get_optimizer = lambda m: Adam(m.parameters())


def fed_avg_test():
    torch.manual_seed(random_seed)
    server = FedAvgServer(
        lambda: Convert.model_to_device(SimpleDNN(784, 128, 10)),
        FedAvgServerOptions(n_clients_per_round=10)
    )

    clients = []
    client_model = Convert.model_to_device(SimpleDNN(784, 128, 10))
    for i in range(n_clients):
        client = FedAvgClient(
            lambda: client_model,
            server,
            FedAvgClientOptions(
                client_data_loader=DataLoader(iid_mnist_datasets[i]),
                get_optimizer=get_optimizer,
                loss_func=loss_func,
                batch_mode=True,
                n_local_rounds=n_local_rounds
            )
        )
        clients.append(client)
    server.set_clients(clients)

    fed_train = FedTrain(server)

    fed_train.start()

    test_records = fed_train.train(
        n_global_rounds=n_global_rounds,
        test_per_global_rounds=1,
        test_data_loader=DataLoader(mnist_test, batch_size=128),
        test_metrics=[multiclass_acc]
    )
    return test_records


def stc_test():
    torch.manual_seed(random_seed)
    server = STCServer(
        lambda: Convert.model_to_device(SimpleDNN(784, 128, 10)),
        BaseCompressionServerOptions(
            n_clients_per_round=10,
            compressor=STC(1/400)
        )
    )

    clients = []
    client_model = Convert.model_to_device(SimpleDNN(784, 128, 10))
    for i in range(n_clients):
        client = STCClient(
            lambda: client_model,
            server,
            BaseCompressionClientOptions(
                client_data_loader=DataLoader(iid_mnist_datasets[i]),
                get_optimizer=get_optimizer,
                loss_func=loss_func,
                batch_mode=True,
                n_local_rounds=n_local_rounds,
                compressor=STC(1/400)
            )
        )
        clients.append(client)
    server.set_clients(clients)


    fed_train = FedTrain(server)

    fed_train.start()

    test_records = fed_train.train(
        n_global_rounds=n_global_rounds,
        test_per_global_rounds=1,
        test_data_loader=DataLoader(mnist_test, batch_size=128),
        test_metrics=[multiclass_acc]
    )
    return test_records


def learned_stc_test(lambda_l1: float, smooth_l1: bool=False, dynamic_l1: bool=False):
    torch.manual_seed(random_seed)

    def ref_metric(pred_ys, ys):
        return -loss_func(Convert.to_tensor(pred_ys), Convert.to_tensor(ys)).item()

    server = LearnedSTCServer(
        lambda: Convert.model_to_device(SimpleDNN(784, 128, 10)),
        LearnedSTCServerOptions(
            n_clients_per_round=10,
            compressor=STC(1/400),
            residual_shrinkage=0.99,
            dynamic_shrinkage=True,
            ref_data_loader=DataLoader(mnist_val, 128),
            ref_metric=ref_metric
        )
    )

    clients = []
    client_model = Convert.model_to_device(SimpleDNN(784, 128, 10))
    for i in range(n_clients):
        client = LearnedSTCClient(
            lambda: client_model,
            server,
            LearnedSTCClientOptions(
                client_data_loader=DataLoader(iid_mnist_datasets[i]),
                get_optimizer=get_optimizer,
                loss_func=loss_func,
                batch_mode=True,
                n_local_rounds=n_local_rounds,
                residual_shrinkage=0.99,
                compressor=STC(1/400),
                lambda_l1=lambda_l1,
                smooth_l1=smooth_l1,
                dynamic_l1=dynamic_l1
            )
        )
        clients.append(client)
    server.set_clients(clients)

    fed_train = FedTrain(server)

    fed_train.start()

    test_records = fed_train.train(
        n_global_rounds=n_global_rounds,
        test_per_global_rounds=1,
        test_data_loader=DataLoader(mnist_test, batch_size=128),
        test_metrics=[multiclass_acc]
    )
    return test_records


if __name__ == '__main__':

    stc_test()

    learned_stc_kwargs = [
        {"lambda_l1": 0},
        {"lambda_l1": 2e-4},
        {"lambda_l1": 2e-4, "smooth_l1": True},
        {"lambda_l1": 1e-3, "smooth_l1": True, "dynamic_l1": True}
    ]


    def prettify_kwargs(kwargs: dict):
        return "Paras:" + "-".join(k + ":" + str(kwargs[k]) for k in kwargs)

    learned_stc_funcs = [lambda kws=kws: learned_stc_test(**kws) for kws in learned_stc_kwargs]

    records = parallel_process([fed_avg_test, stc_test] + learned_stc_funcs,
                               ["cuda:0", "cuda:0", "cuda:0", "cuda:1", "cuda:1", "cuda:1"])

    records = [np.array(record) for record in records]

    record_df = pd.DataFrame(index=records[0][:, 0])
    record_df['fedavg'] = records[0][:, 1]
    record_df['stc'] = records[1][:, 1]
    for kws, record in zip(learned_stc_kwargs, records[2:]):
        record_df['learned_stc-' + prettify_kwargs(kws)] = record[:, 1]
    record_df.to_csv("records_mnist_dnn.csv")
