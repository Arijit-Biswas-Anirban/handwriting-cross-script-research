from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from handwriting_cross_script_research.preprocessing import load_image_tensor


LANGUAGE_TO_LABEL = {
    "Arabic": 0,
    "English": 1,
}


class QUWIDataset(Dataset):
    required_columns = {
        "filename",
        "writer",
        "page_id",
        "language",
        "same_text",
        "experiment_split",
    }

    def __init__(
        self,
        metadata: pd.DataFrame | str | Path,
        image_dir: str | Path,
        target_size: tuple[int, int] = (384, 384),
        experiment_split: str | None = None,
    ):
        if isinstance(metadata, (str, Path)):
            metadata = pd.read_csv(metadata)

        missing_columns = self.required_columns - set(metadata.columns)

        if missing_columns:
            raise ValueError(
                f"Missing metadata columns: {sorted(missing_columns)}"
            )

        if experiment_split is not None:
            metadata = metadata[
                metadata["experiment_split"] == experiment_split
            ]

        if metadata.empty:
            raise ValueError("Dataset metadata is empty")

        self.metadata = metadata.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.target_size = target_size

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.metadata.iloc[index]
        image_path = self.image_dir / row["filename"]

        image_tensor, _ = load_image_tensor(
            image_path,
            target_size=self.target_size,
        )

        return {
            "image": image_tensor,
            "writer_id": torch.tensor(
                int(row["writer"]),
                dtype=torch.long,
            ),
            "page_id": torch.tensor(
                int(row["page_id"]),
                dtype=torch.long,
            ),
            "language_label": torch.tensor(
                LANGUAGE_TO_LABEL[row["language"]],
                dtype=torch.long,
            ),
            "same_text": torch.tensor(
                int(row["same_text"]),
                dtype=torch.long,
            ),
            "filename": row["filename"],
            "language": row["language"],
            "experiment_split": row["experiment_split"],
        }
