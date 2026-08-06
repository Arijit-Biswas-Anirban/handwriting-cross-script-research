from pathlib import Path

import torch
from torch.utils.data import DataLoader

from handwriting_cross_script_research.dataset import QUWIDataset


def create_quwi_dataloaders(
    metadata_path: str | Path,
    image_dir: str | Path,
    target_size: tuple[int, int] = (384, 384),
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool | None = None,
) -> tuple[
    dict[str, QUWIDataset],
    dict[str, DataLoader],
]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    split_names = {
        "train": "development_train",
        "validation": "validation",
        "test": "official_test",
    }

    datasets = {
        name: QUWIDataset(
            metadata=metadata_path,
            image_dir=image_dir,
            target_size=target_size,
            experiment_split=split_name,
        )
        for name, split_name in split_names.items()
    }

    generator = torch.Generator()
    generator.manual_seed(seed)

    common_settings = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "drop_last": False,
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            shuffle=True,
            generator=generator,
            **common_settings,
        ),
        "validation": DataLoader(
            datasets["validation"],
            shuffle=False,
            **common_settings,
        ),
        "test": DataLoader(
            datasets["test"],
            shuffle=False,
            **common_settings,
        ),
    }

    return datasets, loaders
