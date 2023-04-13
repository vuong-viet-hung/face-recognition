import re
from pathlib import Path

from sklearn.model_selection import train_test_split


def identity_from_name(image_name: str) -> str:
    return re.fullmatch(
        r"(\d+)@N(\d{2})_identity_(\d+)@(\d+)_(\d+).jpg", image_name
    ).group(1)


def find_unique_identities(image_paths: list[Path]) -> list[str]:
    return list({identity_from_name(path.name) for path in image_paths})


def split_identities(
    identities: list[str],
    valid_size: float,
    test_size: float,
    random_state: int | None = None,
) -> tuple[list[str], list[str], list[str]]:
    train_identities, test_identities = train_test_split(
        identities, test_size=test_size, random_state=random_state
    )
    train_identities, valid_identities = train_test_split(
        train_identities,
        test_size=valid_size / (1 - test_size),
        random_state=random_state,
    )
    return train_identities, test_identities, valid_identities


def find_paths_with_identities(
    image_paths: list[Path], identities: list[str]
) -> list[Path]:
    identity_set = set(identities)
    return [
        path for path in image_paths
        if identity_from_name(path.name) in identity_set
    ]
