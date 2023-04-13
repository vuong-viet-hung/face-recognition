import torch
import torch.utils.data


def one_hot_encode(numerical_label: torch.Tensor, n_classes) -> torch.Tensor:
    return torch.nn.functional.one_hot(numerical_label, n_classes)


def one_hot_decode(one_hot_label: torch.Tensor) -> torch.Tensor:
    return one_hot_label.argmax(dim=-1).type(torch.long)
