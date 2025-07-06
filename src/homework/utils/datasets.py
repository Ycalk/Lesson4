from torch.utils.data import Dataset
from typing import NamedTuple
import torchvision
import torchvision.transforms as transforms


MergeDataset = NamedTuple(
    "MergeDataset", [("train_dataset", Dataset), ("val_dataset", Dataset)]
)


class MNISTDataset(Dataset):
    @classmethod
    def get_default_transform(cls) -> transforms.Compose:
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    @classmethod
    def get_merged_dataset(
        cls,
        train_transform: transforms.Compose | None = None,
        val_transform: transforms.Compose | None = None,
    ) -> MergeDataset:
        """Объединяет тренировочный и валидационный датасеты MNIST.

        Args:
            train_transform (transforms.Compose | None, optional): Трансформации для тренировочного датасета. Defaults to None.
            val_transform (transforms.Compose | None, optional): Трансформации для валидационного датасета. Defaults to None.

        Returns:
            MergeDataset: Объединенный датасет, содержащий тренировочный и валидационный наборы данных MNIST.
        """
        return MergeDataset(
            train_dataset=cls(
                train=True, transform=train_transform or cls.get_default_transform()
            ),
            val_dataset=cls(
                train=False, transform=val_transform or cls.get_default_transform()
            ),
        )

    def __init__(self, train=True, transform=None):
        super().__init__()
        self.dataset = torchvision.datasets.MNIST(
            root="./data", train=train, download=True, transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class CIFARDataset(Dataset):
    @classmethod
    def get_default_transform(cls) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    @classmethod
    def get_merged_dataset(
        cls,
        train_transform: transforms.Compose | None = None,
        val_transform: transforms.Compose | None = None,
    ) -> MergeDataset:
        """Объединяет тренировочный и валидационный датасеты CIFAR-10.

        Args:
            train_transform (transforms.Compose | None, optional): Трансформации для тренировочного датасета. Defaults to None.
            val_transform (transforms.Compose | None, optional): Трансформации для валидационного датасета. Defaults to None.

        Returns:
            MergeDataset: Объединенный датасет, содержащий тренировочный и валидационный наборы данных CIFAR-10.
        """
        return MergeDataset(
            train_dataset=cls(
                train=True, transform=train_transform or cls.get_default_transform()
            ),
            val_dataset=cls(
                train=False, transform=val_transform or cls.get_default_transform()
            ),
        )

    def __init__(self, train=True, transform=None):
        super().__init__()
        self.dataset = torchvision.datasets.CIFAR10(
            root="./data", train=train, download=True, transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]