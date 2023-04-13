from abc import ABC, abstractmethod
from typing import Tuple

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from face_recognition.operators import one_hot_decode


class Metric(ABC):

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def __call__(
        self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> float:
        pass

    @abstractmethod
    def update(
        self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        pass

    @abstractmethod
    def result(self) -> float:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def merge(self, other) -> None:
        pass


class Accuracy(Metric):

    def __init__(self) -> None:
        self.prediction_count = 0
        self.correct_prediction_count = 0
        super().__init__("accuracy")

    def __call__(
        self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> float:
        n_predictions, *_ = output_batch.shape
        n_correct_predictions = (
            one_hot_decode(output_batch) == one_hot_decode(target_batch)
        ).sum().item()
        return n_correct_predictions / n_predictions

    def update(
        self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        batch_size, *_ = output_batch.shape
        self.prediction_count += batch_size
        self.correct_prediction_count += (
            one_hot_decode(output_batch) == one_hot_decode(target_batch)
        ).sum().item()

    def result(self) -> float:
        return self.correct_prediction_count / self.prediction_count

    def reset(self) -> None:
        self.prediction_count = 0
        self.correct_prediction_count = 0

    def merge(self, other) -> None:
        self.prediction_count += other.prediction_count
        self.correct_prediction_count += other.correct_prediction_count


class AveragePrecision(Metric):

    def __init__(self) -> None:
        self.total_ap = 0.0
        self.prediction_count = 0
        super().__init__("ap")

    def __call__(
        self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> float:
        labels, predictions = compare_embeddings(output_batch, target_batch)
        return average_precision_score(labels, predictions)

    def update(
        self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        batch_aur = self(output_batch, target_batch)
        n_predictions, _ = batch_aur
        self.total_ap += batch_aur * n_predictions
        self.prediction_count += n_predictions

    def result(self) -> float:
        return self.total_ap / self.prediction_count

    def reset(self) -> None:
        self.total_ap = 0.0
        self.prediction_count = 0

    def merge(self, other) -> None:
        self.total_ap += other.total_ap
        self.prediction_count += other.prediction_count


class AUR(Metric):

    def __init__(self) -> None:
        self.total_aur = 0.0
        self.prediction_count = 0
        super().__init__("aur")

    def __call__(
        self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> float:
        labels, predictions = compare_embeddings(output_batch, target_batch)
        return roc_auc_score(labels, predictions)

    def update(
        self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        batch_aur = self(output_batch, target_batch)
        n_predictions, _ = batch_aur
        self.total_aur += batch_aur * n_predictions
        self.prediction_count += n_predictions

    def result(self) -> float:
        return self.total_aur / self.prediction_count

    def reset(self) -> None:
        self.total_aur = 0.0
        self.prediction_count = 0

    def merge(self, other) -> None:
        self.total_aur += other.total_ap
        self.prediction_count += other.prediction_count


def compare_embeddings(
    embeddings: torch.Tensor, targets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    predictions = []
    labels = []

    for idx, (embedding, target) in enumerate(zip(embeddings, targets)):
        if idx + 1 == len(embeddings):
            break
        cosine_similarity = _cosine_similarity(embedding, embeddings[idx + 1:])
        labels.extend(_cosine_similarity(target, targets[idx + 1:]))
        predictions.extend(_cos_to_prediction(cosine_similarity))

    predictions = torch.vstack(predictions).detach().cpu()
    labels = torch.vstack(labels).detach().cpu()

    return labels, predictions


def _cosine_similarity(
    key_embedding: torch.Tensor, query_embeddings: torch.Tensor
) -> torch.Tensor:
    return (key_embedding * query_embeddings).sum(axis=1).clamp(-1, 1)


def _cos_to_prediction(cosine_similarity: torch.Tensor) -> torch.Tensor:
    return 1 - cosine_similarity.acos() / torch.pi
