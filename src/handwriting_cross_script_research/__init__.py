from handwriting_cross_script_research.dataloader import create_quwi_dataloaders
from handwriting_cross_script_research.dataset import LANGUAGE_TO_LABEL, QUWIDataset
from handwriting_cross_script_research.preprocessing import (
    load_image_tensor,
    resize_with_padding,
)

__all__ = [
    "LANGUAGE_TO_LABEL",
    "QUWIDataset",
    "create_quwi_dataloaders",
    "load_image_tensor",
    "resize_with_padding",
]


def main() -> None:
    print("Handwriting cross-script research package is ready.")
