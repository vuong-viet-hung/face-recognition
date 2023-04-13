from pathlib import Path
from typing import List, Literal, Optional, Union

import torch.utils.data

from face_recognition.models import ArcFace
from face_recognition.losses import Loss
from face_recognition.metrics import Metric
from face_recognition.monitors import Monitor


class ArcFaceTrainer:

    def __init__(
        self,
        embedder: torch.nn.Module,
        arcface: ArcFace,
        loss: Loss,
        optimizer: torch.optim.Optimizer,
        metrics: Optional[List[Metric]] = None,
        monitors: Optional[List[Monitor]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        self._embedder = embedder.to(device)
        self._arcface = arcface.to(device)
        self._loss = loss
        self._optimizer = optimizer
        self._metrics = [] if metrics is None else metrics
        self._monitors = [] if monitors is None else monitors
        self._device = device
        self._tqdm, self._trange = _import_tqdm()
        self.running = True

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        n_epochs: int,
    ):
        progress_bar = self._trange(n_epochs)

        for epoch in progress_bar:

            progress_bar.set_description(
                f"Epoch {epoch + 1}/{len(progress_bar)}: "
                f"lr = {self._optimizer.param_groups[0]['lr']:.2e}"
            )
            self._train_one_epoch(train_loader)
            self._valid_one_epoch(valid_loader)

            if not self.running:
                break

    def _train_one_epoch(
            self, train_loader: torch.utils.data.DataLoader
    ) -> None:
        progress_bar = self._tqdm(train_loader)
        self._train_mode()
        for input_batch, target_batch in progress_bar:
            self._train_one_step(input_batch, target_batch)
            progress_bar.set_description(
                f"Train: loss = {self._loss.result():.4f}, {self._format_metric_results()}"
            )

        self._notify_monitors("train")
        self._reset()

    def _train_one_step(
            self, input_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        input_batch = input_batch.to(self._device)
        target_batch = target_batch.to(self._device)

        embedding_batch = self._embedder(input_batch)
        output_batch = self._arcface(embedding_batch, target_batch)
        computed_loss = self._loss(output_batch, target_batch)

        computed_loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()

        self._loss.update(output_batch, target_batch)
        self._update_metrics(embedding_batch, target_batch)

    def _valid_one_epoch(
        self, valid_loader: torch.utils.data.DataLoader
    ) -> None:
        progress_bar = self._tqdm(valid_loader)
        self._valid_mode()

        with torch.no_grad():
            for input_batch, target_batch in progress_bar:
                self._valid_one_step(input_batch, target_batch)
                progress_bar.set_description(f"Valid: {self._format_metric_results()}")

        self._notify_monitors("valid")
        self._reset()

    def _valid_one_step(
            self, input_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        input_batch = input_batch.to(self._device)
        target_batch = target_batch.to(self._device)
        embedding_batch = self._embedder(input_batch)

        self._update_metrics(embedding_batch, target_batch)

    def test(self, test_loader: torch.utils.data.DataLoader) -> None:
        progress_bar = self._tqdm(test_loader)
        self._valid_mode()

        with torch.no_grad():
            for input_batch, target_batch in progress_bar:
                self._valid_one_step(input_batch, target_batch)
                progress_bar.set_description(f"Test: {self._format_metric_results()}")

        self._reset()

    def save(self, save_dir: Union[str, Path]) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_embedder_path = save_dir / "embedder.pth"
        torch.save(self._embedder.state_dict(), saved_embedder_path)

        saved_arcface_path = save_dir / "arcface.pth"
        torch.save(self._arcface.state_dict(), saved_arcface_path)

        saved_optimizer_path = save_dir / "optimizer.pth"
        torch.save(self._optimizer.state_dict(), saved_optimizer_path)

    def load(self, save_dir: Union[str, Path]) -> None:
        save_dir = Path(save_dir)

        saved_embedder_path = save_dir / "embedder.pth"
        self._embedder.load_state_dict(torch.load(saved_embedder_path))

        saved_arcface_path = save_dir / "arcface.pth"
        self._arcface.load_state_dict(torch.load(saved_arcface_path))

        saved_optimizer_path = save_dir / "optimizer.pth"
        self._optimizer.load_state_dict(torch.load(saved_optimizer_path))

    def _update_metrics(
        self, output_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> None:
        for metric in self._metrics:
            metric.update(output_batch, target_batch)

    def _reset(self) -> None:
        self._loss.reset()
        for metric in self._metrics:
            metric.reset()

    def _format_metric_results(self) -> str:
        results_format = ", ".join(
            f"{metric.name} = {metric.result():.4f}" for metric in self._metrics
        )
        return results_format

    def _notify_monitors(self, phase: str):
        for monitor in self._monitors:
            monitor.update(phase, self)

    def _train_mode(self) -> None:
        self._embedder.train()
        self._arcface.train()

    def _valid_mode(self) -> None:
        self._embedder.eval()
        self._arcface.eval()


def _import_tqdm() -> tuple:
    from tqdm.auto import tqdm, trange
    return tqdm, trange
