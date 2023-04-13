from abc import ABC, abstractmethod

import torch

from face_recognition.operators import one_hot_decode


class Loss(ABC):
    def __init__(self) -> None:
        self.total_loss: float = 0.0
        self.sample_count: int = 0

    @abstractmethod
    def __call__(
        self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> torch.Tensor:
        pass

    def update(self, output_batch: torch.Tensor, target_batch: torch.Tensor) -> None:
        batch_size, *_ = output_batch.shape
        batch_loss = self(output_batch, target_batch).item()
        self.sample_count += batch_size
        self.total_loss += batch_loss * batch_size

    def result(self) -> float:
        return self.total_loss / self.sample_count

    def reset(self) -> None:
        self.total_loss = 0.0
        self.sample_count = 0

    def merge(self, other) -> None:
        self.total_loss += other.total_loss
        self.sample_count += other.sample_count


class CrossEntropy(Loss):
    def __init__(self) -> None:
        super().__init__()
        self._loss_fn = torch.nn.CrossEntropyLoss()

    def __call__(
        self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> torch.Tensor:
    	print(output_batch[0], target_batch[0])
    	print(
    	    self._loss_fn(
                output_batch,
                one_hot_decode(target_batch),
            )
        )
        return self._loss_fn(
            output_batch,
            one_hot_decode(target_batch),
        )

