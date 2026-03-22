from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import crop, resized_crop, to_pil_image
from transformers import AutoModelForImageSegmentation

from .settings import BackgroundRemovalConfig
from config.settings import ModelVersionsConfig
from logger_config import logger

if TYPE_CHECKING:
    from ben2 import BEN_Base


class BackgroundRemovalService(ABC):
    def __init__(self, settings: BackgroundRemovalConfig, model_versions: ModelVersionsConfig):
        """
        Initialize the BackgroundRemovalService.
        """
        self.settings = settings
        self.model_versions = model_versions

        # Set padding percentage and output size for centering and resizing
        self.padding_percentage = self.settings.padding_percentage
        self.limit_padding = self.settings.limit_padding

        # Set output size (if None, no resizing after cropping)
        self.output_size = self.settings.output_image_size

        # Set device
        self.device = f"cuda:{settings.gpu}" if torch.cuda.is_available() else "cpu"

        self.model, self.transforms = self._initialize_model_and_transforms()

        # Set normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    async def startup(self) -> None:
        """
        Startup the BackgroundRemovalService.
        """
        logger.info(f"Loading {self.settings.model_id} model...")

        # Load model
        try:
            self.model = self._load_model()
            logger.success(f"{self.settings.model_id} model loaded.")
        except Exception as e:
            logger.error(f"Error loading {self.settings.model_id} model: {e}")
            raise RuntimeError(f"Error loading {self.settings.model_id} model: {e}")

    async def shutdown(self) -> None:
        """
        Shutdown the BackgroundRemovalService.
        """
        self.model = None
        logger.info("BackgroundRemovalService closed.")

    def ensure_ready(self) -> None:
        """
        Ensure the BackgroundRemovalService is ready.
        """
        if self.model is None:
            raise RuntimeError(f"{self.settings.model_id} model not initialized.")

    def remove_background(self, image: Image.Image | Iterable[Image.Image]) -> Image.Image | Iterable[Image.Image]:
        """
        Remove the background from the image.
        """
        # try:
        t1 = time.time()

        images = image if isinstance(image, Iterable) else [image]

        outputs = []
        has_alpha = False

        for img in images:
            if img.mode == "RGBA":
                # Get alpha channel
                alpha = np.array(img)[:, :, 3]
                if not np.all(alpha==255):
                    has_alpha=True
            
            if has_alpha:
                # If the image has alpha channel, return the image
                output = img
                
            else:
                # PIL.Image (H, W, C) C=3
                # Tensor (H, W, C) -> (C, H',W')
                # rgb_tensor = self.transforms(rgb_image).to(self.device)
                tensor_rgb, mask = self._remove_background(img)
                output = self._crop_and_center(tensor_rgb, mask)

            outputs.append(output)

        images_without_background = tuple(to_pil_image(o[:3]) for o in outputs) if isinstance(image, Iterable) else to_pil_image(outputs[0][:3])

        removal_time = time.time() - t1
        logger.success(f"Background remove - Time: {removal_time:.2f}s - Images without background: {len(images_without_background)}")

        return images_without_background

    def _crop_and_center(self, tensor_rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Remove the background from the image.
        """

        # Normalize tensor value for background removal model, reshape for model batch processing (C=3, H, W) -> (1, C=3, H, W)

        # Get bounding box indices
        bbox_indices = torch.argwhere(mask > 0.8)
        if len(bbox_indices) == 0:
            crop_args = dict(top = 0, left = 0, height = mask.shape[1], width = mask.shape[0])
        else:
            h_min, h_max = torch.aminmax(bbox_indices[:, 1])
            w_min, w_max = torch.aminmax(bbox_indices[:, 0])
            width, height = w_max - w_min, h_max - h_min
            center =  (h_max + h_min) / 2, (w_max + w_min) / 2
            size = max(width, height)
            padded_size_factor = 1 + self.padding_percentage
            size = int(size * padded_size_factor)
            size = max(size, 2)  # Ensure size is at least 2 to avoid zero-size crop

            top = int(center[1] - size // 2)
            left = int(center[0] - size // 2)
            bottom = int(center[1] + size // 2)
            right = int(center[0] + size // 2)

            if self.limit_padding:
                top = max(0, top)
                left = max(0, left)
                bottom = min(mask.shape[1], bottom)
                right = min(mask.shape[0], right)
            
            crop_args = dict(
                top=top,
                left=left,
                height=bottom - top,
                width=right - left
            )
        

        mask = mask.unsqueeze(0)
        # Concat mask with image and blacken the background: (C=3, H, W) | (1, H, W) -> (C=4, H, W)
        tensor_rgba = torch.cat([tensor_rgb*mask, mask], dim=-3)
        output = resized_crop(tensor_rgba, **crop_args, size = self.output_size, antialias=False) if self.output_size else crop(tensor_rgba, **crop_args)
        return output

    @abstractmethod
    def _initialize_model_and_transforms(self) -> tuple[AutoModelForImageSegmentation | BEN_Base, transforms.Compose]:
        """
        Initialize model and transforms.
        """
        pass

    @abstractmethod
    def _load_model(self) -> AutoModelForImageSegmentation | BEN_Base:
        """
        Load the background removal model.
        """
        pass
    @abstractmethod
    def _remove_background(self, image: Image):
        """
        Remove the background from the image.
        """
        pass