import copy
from abc import ABC, abstractmethod
from typing import Dict, List

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


class ResultsByID:

    def __init__(self) -> None:
        self._n_identities = 0
        self.n_updates = 0
        self._embeddings_by_id = []
        self._results_by_id = []

    def update(
        self, embedding_batch: torch.Tensor, label_batch: torch.Tensor
    ) -> None:
        _, n_identities = label_batch.shape
        class_indices = one_hot_decode(label_batch)

        if self._n_identities == 0:
            self._change_phase(n_identities)

        for idx, embedding in zip(class_indices, embedding_batch):
            self._embeddings_by_id[idx].append(embedding)

        for key_idx, key_id_embeddings in enumerate(self._embeddings_by_id):
            query_id_embeddings = sum(
                [
                    embeddings for query_idx, embeddings
                    in enumerate(self._embeddings_by_id) if query_idx != key_idx
                ],
                start=[],
            )
            key_id_query_embeddings = copy.copy(key_id_embeddings)

            while len(key_id_query_embeddings) > 0:
                key_embedding = key_id_query_embeddings.pop(0)
                query_embeddings = key_id_query_embeddings + query_id_embeddings
                self._results_by_id[key_idx]["predictions"].extend(
                    _cosine_similarity(key_embedding, query_embeddings)
                )
                labels = (
                    [torch.tensor(1) for _ in key_id_query_embeddings]
                    + [torch.tensor(0) for _ in query_id_embeddings]
                )
                self._results_by_id[key_idx]["labels"].extend(labels)

        self.n_updates += 1

    def result(self) -> List[Dict[str, torch.Tensor]]:
        return [
            {
                "predictions": torch.tensor(result_dict["predictions"]).detach().cpu(),
                "labels": torch.tensor(result_dict["labels"]).detach().cpu(),
            } for result_dict in self._results_by_id
        ]

    def reset(self) -> None:
        self._embeddings_by_id = []
        self._results_by_id = []
        self._n_identities = 0

    def _change_phase(self, n_identities: int) -> None:
        self._n_identities = n_identities
        self._embeddings_by_id = [[] for _ in range(n_identities)]
        self._results_by_id = [
            {"predictions": [], "labels": []} for _ in range(n_identities)
        ]


def _cosine_similarity(
    key_embedding: torch.Tensor, query_embeddings: List[torch.Tensor]
) -> torch.Tensor:
    return (key_embedding * torch.vstack(query_embeddings)).sum(axis=1)


class AveragePrecision(Metric):

    def __init__(self, result_by_id: ResultsByID) -> None:
        self._results_by_id = result_by_id
        self._n_updates = 0
        super().__init__("average_precision")

    def __call__(
        self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> float:
        pass

    def update(
        self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        if self._n_updates == self._results_by_id.n_updates:
            self._results_by_id.update(output_batch, target_batch)
        self._n_updates += 1

    def result(self) -> float:
        precisions = torch.tensor(
            [
                average_precision_score(
                    result_dict["labels"], result_dict["predictions"]
                ) for result_dict in self._results_by_id.result()
            ]
        )
        return precisions.mean().item()

    def reset(self) -> None:
        self._results_by_id.reset()

    def merge(self, other) -> None:
        pass


class AUR(Metric):

    def __init__(self, result_by_id: ResultsByID) -> None:
        self._results_by_id = result_by_id
        self._n_updates = 0
        super().__init__("aur")

    def __call__(
            self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> float:
        pass

    def update(
            self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        if self._n_updates == self._results_by_id.n_updates:
            self._results_by_id.update(output_batch, target_batch)
        self._n_updates += 1

    def result(self) -> float:
        aur = torch.tensor(
            [
                roc_auc_score(
                    result_dict["labels"], result_dict["predictions"]
                ) for result_dict in self._results_by_id.result()
            ]
        )
        return aur.mean().item()

    def reset(self) -> None:
        self._results_by_id.reset()

    def merge(self, other) -> None:
        pass
