import torch


class ArcFace(torch.nn.Module):

    def __init__(
        self,
        n_identities: int,
        n_embeddings: int = 512,
        n_subclasses: int = 1,
        margin=0.5,
        scale_factor=64,
    ) -> None:
        super().__init__()
        self._w = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.zeros(n_embeddings, n_identities, n_subclasses)
            )
        )
        self._margin = margin
        self._scale_factor = scale_factor
        self._softmax = torch.nn.LogSoftmax(dim=1)

    def forward(
        self,
        embedding_batch: torch.Tensor,
        target_batch: torch.Tensor,
    ) -> torch.Tensor:
        normalized_w = torch.nn.functional.normalize(self._w)
        cosine_similarities_per_sub_classes = _cosine_similarity(
            embedding_batch, normalized_w
        )
        cosine_similarities = cosine_similarities_per_sub_classes.max(dim=2).values
        theta = cosine_similarities.acos()
        margins = target_batch * self._margin
        return self._softmax(
            (theta + margins).cos() * self._scale_factor
        )


def _cosine_similarity(embedding_batch: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (embedding_batch[:, :, None, None] * w).sum(dim=1)
