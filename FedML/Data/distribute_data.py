import numpy as np


def get_iid_mnist(mnist_np: np.ndarray, samples_per_client: int):
    """
    Get identically distributed mnist datasets
    i.e. Each client will have the same distribution of data
    This is done by simply random distribute the samples
    :param mnist_np: MNIST dataset in numpy format, shape [N,. 794]. N: number of samples; 794 = 784 (x) + 10 (y)
    :param samples_per_client:
    :return:
    """
    client_datasets = []
    for i in range(int(mnist_np.shape[0] / samples_per_client)):
        client_datasets.append(mnist_np[i * samples_per_client: (i + 1) * samples_per_client])
    return client_datasets


def get_non_iid_mnist(mnist_np: np.ndarray, classes_per_client: int, samples_per_class: int):
    """
    Get non-identically distributed mnist dataset
    Each client will have data of several digits
    :param mnist_np:
    :param classes_per_client:
    :param samples_per_class:
    :return:
    """
    class_counts = [int(5000 / samples_per_class) for _ in range(10)]
    mnist_classified = [mnist_np[mnist_np[:, 0] == i] for i in range(10)]
    client_datasets = []
    while sum(class_counts) >= 2:
        i = np.random.randint(10)
        client_classes = []
        while len(client_classes) != 2:
            if class_counts[i] != 0:
                j = class_counts[i]
                client_classes.append(mnist_classified[i][(j - 1) * classes_per_client: j * classes_per_client])
                class_counts[i] -= 1
            i = (i + 1) % 10
        client_datasets.append(np.vstack(client_classes))
    return client_datasets
