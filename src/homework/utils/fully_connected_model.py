import torch
from torch import nn
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Layer(ABC):
    @abstractmethod
    def construct(self, input_dim: int) -> nn.Module:
        """Создает слой нейронной сети.

        Args:
            input_dim (int): Размерность входных данных.

        Returns:
            nn.Module: Слой нейронной сети, готовый к использованию.
        """
        ...

    @property
    def output_dim(self) -> int | None:
        """Получает размерность выходных данных слоя.

        Returns:
            (int | None): Размерность выходных данных слоя, или None, если размерность входных данных не определена.
        """
        return None


@dataclass(frozen=True)
class LinearLayer(Layer):
    size: int

    def construct(self, input_dim: int):
        return nn.Linear(input_dim, self.size)

    @property
    def output_dim(self) -> int:
        return self.size


@dataclass(frozen=True)
class ReLULayer(Layer):
    def construct(self, _: int):
        return nn.ReLU()


@dataclass(frozen=True)
class Sigmoid(Layer):
    def construct(self, _: int):
        return nn.Sigmoid()


@dataclass(frozen=True)
class Dropout(Layer):
    rate: float = 0.5

    def construct(self, _: int):
        return nn.Dropout(self.rate)


@dataclass(frozen=True)
class BatchNorm(Layer):
    def construct(self, input_dim: int):
        return nn.BatchNorm1d(input_dim)


@dataclass(frozen=True)
class LayerNorm(Layer):
    def construct(self, input_dim: int):
        return nn.LayerNorm(input_dim)


class FullyConnectedModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layers: list[Layer]) -> None:
        """Инициализация модели нейронной сети.

        Args:
            input_dim (int): Размерность входных данных.
            output_dim (int): Размерность выходных данных.
            layers (list[Layer]): Список слоев, которые будут использоваться в модели.
        """
        super().__init__()

        self.layers: list[nn.Module] = []

        prev_size = input_dim
        for layer in layers:
            layer_module = layer.construct(prev_size)
            self.layers.append(layer_module)
            prev_size = layer.output_dim if layer.output_dim else prev_size
        self.layers.append(nn.Linear(prev_size, output_dim))

        self.network = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.network(x)


if __name__ == "__main__":
    # Пример использования
    model = FullyConnectedModel(
        input_dim=784,
        output_dim=10,
        layers=[
            LinearLayer(size=128),
            ReLULayer(),
            Dropout(rate=0.5),
            BatchNorm(),
            LinearLayer(size=64),
            ReLULayer(),
            BatchNorm(),
            LinearLayer(size=32),
            Sigmoid(),
        ],
    )
    model.eval()
    x = torch.randn(16, 784)
    assert model(x).shape == torch.Size([16, 10]), (
        "Выходная размерность должна быть [16, 10]"
    )