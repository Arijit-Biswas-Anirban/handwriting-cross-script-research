from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF


def resize_with_padding(
    image: Image.Image,
    target_size: tuple[int, int] = (384, 384),
    output_mode: str = "RGB",
    fill_value: int = 255,
) -> tuple[Image.Image, dict[str, object]]:
    if len(target_size) != 2 or any(size <= 0 for size in target_size):
        raise ValueError("target_size must contain two positive integers")

    original_size = image.size
    converted_image = image.convert(output_mode)

    resized_image = ImageOps.contain(
        converted_image,
        target_size,
        method=Image.Resampling.LANCZOS,
    )

    fill = (
        (fill_value, fill_value, fill_value)
        if output_mode == "RGB"
        else fill_value
    )

    canvas = Image.new(
        output_mode,
        target_size,
        fill,
    )

    left = (target_size[0] - resized_image.width) // 2
    top = (target_size[1] - resized_image.height) // 2

    canvas.paste(
        resized_image,
        (left, top),
    )

    padding = {
        "left": left,
        "top": top,
        "right": target_size[0] - resized_image.width - left,
        "bottom": target_size[1] - resized_image.height - top,
    }

    information = {
        "original_size": original_size,
        "resized_size": resized_image.size,
        "output_size": canvas.size,
        "padding": padding,
        "output_mode": output_mode,
    }

    return canvas, information


def load_image_tensor(
    image_path: str | Path,
    target_size: tuple[int, int] = (384, 384),
) -> tuple[torch.Tensor, dict[str, object]]:
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        processed_image, information = resize_with_padding(
            image,
            target_size=target_size,
            output_mode="RGB",
        )

    tensor = TF.to_tensor(processed_image)

    return tensor, information
