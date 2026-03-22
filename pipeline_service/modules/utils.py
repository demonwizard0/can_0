from PIL import Image

import io
import base64
from datetime import datetime
from typing import Iterable, Optional, Tuple
import os
import random
import numpy as np
import torch

from logger_config import logger
from torchvision.utils import make_grid
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from modules.mesh_generator.schemas import TrellisResult

from config.settings import settings

def secure_randint(low: int, high: int) -> int:
    """ Return a random integer in [low, high] using os.urandom. """
    range_size = high - low + 1
    num_bytes = 4
    max_int = 2**(8 * num_bytes) - 1

    while True:
        rand_bytes = os.urandom(num_bytes)
        rand_int = int.from_bytes(rand_bytes, 'big')
        if rand_int <= max_int - (max_int % range_size):
            return low + (rand_int % range_size)

def set_random_seed(seed: int) -> None:
    """ Function for setting global seed. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

def decode_image(prompt: str) -> Image.Image:
    """
    Decode the image from the base64 string.

    Args:
        prompt: The base64 string of the image.

    Returns:
        The image.
    """
    # Decode the image from the base64 string
    image_bytes = base64.b64decode(prompt)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def to_png_base64(image: Image.Image) -> str:
    """
    Convert the image to PNG format and encode to base64.

    Args:
        image: The image to convert.

    Returns:
        Base64 encoded PNG image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    # Convert to base64 from bytes to string
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def save_file_bytes(data: bytes, folder: str, prefix: str, suffix: str) -> None:
    """
    Save binary data to the output directory.

    Args:
        data: The data to save.
        folder: The folder to save the file to.
        prefix: The prefix of the file.
        suffix: The suffix of the file.
    """
    target_dir = settings.output.output_dir / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    path = target_dir / f"{prefix}_{timestamp}{suffix}"
    try:
        path.write_bytes(data)
        logger.debug(f"Saved file {path}")
    except Exception as exc:
        logger.error(f"Failed to save file {path}: {exc}")

def save_image(image: Image.Image, folder: str, prefix: str, timestamp: str) -> None:
    """
    Save PIL Image to the output directory.

    Args:
        image: The PIL Image to save.
        folder: The folder to save the file to.
        prefix: The prefix of the file.
        timestamp: The timestamp of the file.
    """
    target_dir = settings.output.output_dir / folder / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{prefix}.png"
    try:
        image.save(path, format="PNG")
        logger.debug(f"Saved image {path}")
    except Exception as exc:
        logger.error(f"Failed to save image {path}: {exc}")

def save_files(
    trellis_result: Optional[TrellisResult], 
    image_edited: Image.Image, 
    image_without_background: Image.Image
) -> None:
    """
    Save the generated files to the output directory.

    Args:
        trellis_result: The Trellis result to save.
        image_edited: The edited image to save.
        image_without_background: The image without background to save.
    """
    # Save the Trellis result if available
    if trellis_result:
        if trellis_result.file_bytes:
            format = 'glb'
            save_file_bytes(trellis_result.file_bytes, folder=format, prefix="mesh", suffix=f".{format}")

    # Save the images using PIL Image.save()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    save_image(image_edited, "png", "image_edited", timestamp)
    save_image(image_without_background, "png", "image_without_background", timestamp)

def image_grid(images: Iterable[Image.Image], resize: Optional[Tuple]=(512,512)) -> Image.Image:
    images = images if resize is None else (img.resize(resize) for img in images)
    tensor_grid = make_grid([pil_to_tensor(img) for img in images])
    return to_pil_image(tensor_grid)

