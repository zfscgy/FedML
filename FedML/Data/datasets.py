from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor


data_root = Path("/home/zf/projects/data")


class Mnist:
    @staticmethod
    def get(txs: list = None, tys: list = None):
        """
        :param txs: xs transforms
        :param tys: ys transforms
        :return:
        """
        txs = txs or []
        tys = tys or []
        mnist_train = MNIST((data_root / "mnist").as_posix(),
                            transform=Compose([ToTensor()] + txs), target_transform=Compose(tys))
        mnist_test = MNIST((data_root / "mnist").as_posix(),
                           transform=Compose([ToTensor()] + tys), target_transform=Compose(tys))
        return mnist_train, mnist_test

    @staticmethod
    def tx_flatten(x: torch.Tensor):
        return x.view(28 * 28)

    @staticmethod
    def ty_onehot(y: torch.Tensor):
        return torch.eye(10)[y]


if __name__ == '__main__':
    def mnist_test():
        train, test = Mnist.get([Mnist.tx_flatten], [Mnist.ty_onehot])
        xs, ys = next(iter(DataLoader(train)))
        return xs, ys

    xs, ys = mnist_test()
    pass