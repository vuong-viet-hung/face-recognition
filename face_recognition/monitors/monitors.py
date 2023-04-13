import operator
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Literal, List, Union

import torch
from torch.utils.tensorboard import SummaryWriter

from face_recognition.losses import Loss
from face_recognition.metrics import Metric


Criterion = Union[Loss, Metric]
Comparator = Callable[[float, float], bool]


class Monitor(ABC):
    @abstractmethod
    def update(self, phase: str, model) -> None:
        pass


class EarlyStopping(Monitor):
    def __init__(
        self, criterion: Criterion, patience: int, phase: str = "valid"
    ) -> None:
        self._criterion = criterion
        self._patience = patience
        self._phase = phase
        self._comparator = operator.gt if isinstance(criterion, Loss) else operator.lt
        self._best_result = float("inf" if isinstance(criterion, Loss) else "-inf")
        self._epoch_without_improve = 0

    def update(self, phase: str, model) -> None:
        if phase != self._phase:
            return
        current_result = self._criterion.result()
        if self._comparator(self._best_result, current_result):
            self._best_result = current_result
            self._epoch_without_improve = 0
            return
        self._epoch_without_improve += 1
        if self._epoch_without_improve >= self._patience:
            model.running = False


class ModelCheckpoint(Monitor):
    def __init__(
        self,
        criterion: Criterion,
        checkpoint_dir: Union[str, Path],
        phase: Phase = "valid",
    ) -> None:
        self._criterion = criterion
        self._phase = phase
        self._comparator = operator.gt if isinstance(criterion, Loss) else operator.lt
        self._best_result = float("inf" if isinstance(criterion, Loss) else "-inf")
        self._checkpoint_dir = Path(checkpoint_dir)

    def update(self, phase: str, model) -> None:
        if phase != self._phase:
            return
        current_result = self._criterion.result()
        if self._comparator(self._best_result, current_result):
            self._best_result = current_result
            model.save(self._checkpoint_dir)


class ReduceLROnPlateau(Monitor):
    def __init__(
        self,
        criterion: Criterion,
        optimizer,
        patience: int = 1,
        factor: float = 0.5,
        phase: Phase = "valid",
    ) -> None:
        mode = "min" if isinstance(criterion, Loss) else "max"
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode, factor, patience - 1
        )
        self._criterion = criterion
        self._phase = phase

    def update(self, phase: Phase, model) -> None:
        if phase == self._phase:
            self._scheduler.step(self._criterion.result())


class TensorBoard(Monitor):
    def __init__(
        self,
        criteria: Union[List[Criterion], Dict[str, Criterion]],
        writer: SummaryWriter,
    ) -> None:
        self._criteria = (
            _make_criterion_dict(criteria) if isinstance(criteria, list) else criteria
        )
        self._writer = writer
        self._result_dict: dict[str, dict[Phase, float]] = defaultdict(dict)
        self._epoch = 0

    def update(self, phase: str, model) -> None:
        for name, criterion in self._criteria.items():
            self._result_dict[name][phase] = criterion.result()
        if phase == "train":
            return
        for name, result in self._result_dict.items():
            self._writer.add_scalars(name, result, self._epoch)
        self._epoch += 1


def _make_criterion_dict(criteria: List[Criterion]) -> Dict[str, Criterion]:
    criterion_dict: dict[str, Union[Loss, Metric]] = {}
    for criterion in criteria:
        if isinstance(criterion, Loss):
            criterion_dict["loss"] = criterion
            continue
        criterion_dict[criterion.name] = criterion
    return criterion_dict
