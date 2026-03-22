import math
import time
from typing import Iterable, Optional

import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.models import QwenImageTransformer2DModel
from PIL import Image

from .settings import QwenConfig
from config.settings import ModelVersionsConfig
from logger_config import logger
from modules.image_edit.qwen_manager import QwenManager
from modules.image_edit.prompting import Prompting, TextPrompting

INPUT_IMAGE_SIZE = 1024 * 1024


class QwenEditModule(QwenManager):
    """Qwen module for image editing operations."""

    def __init__(self, settings: QwenConfig, model_versions: ModelVersionsConfig):
        super().__init__(settings, model_versions)
        self._empty_image = self._prepare_input_image(Image.new('RGB', (64, 64)))

        self.base_model_path = settings.base_model_path
        self.edit_model_path = settings.model_path

        self.pipe_config = {
            "num_inference_steps": settings.num_inference_steps,
            "true_cfg_scale": settings.true_cfg_scale,
            "height": settings.height,
            "width": settings.width,

        }

    def _get_model_transformer(self):
        """Load the Nunchaku Qwen transformer for image editing."""
        model_revision = self.model_versions.get_revision(self.edit_model_path)

        return  QwenImageTransformer2DModel.from_pretrained(
                self.edit_model_path,
                subfolder="transformer",
                torch_dtype=self.dtype,
                revision=model_revision
            )

    def _get_model_pipe(self, transformer, scheduler):
        model_revision = self.model_versions.get_revision(self.edit_model_path)

        return QwenImageEditPlusPipeline.from_pretrained(
                self.edit_model_path,
                transformer=transformer,
                scheduler=scheduler,
                torch_dtype=self.dtype,
                revision=model_revision
            )

    def _get_scheduler_config(self):
        """Return scheduler configuration for image editing."""
        return  {
                "base_image_seq_len": 256,
                "base_shift": math.log(3),  # We use shift=3 in distillation
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),  # We use shift=3 in distillation
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,  # set shift_terminal to None
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }

    def _prepare_input_image(self, image: Image, pixels: int = INPUT_IMAGE_SIZE):
        total = int(pixels)

        scale_by = math.sqrt(total / (image.width * image.height))
        width = round(image.width * scale_by)
        height = round(image.height * scale_by)

        return image.resize((width, height), Image.Resampling.LANCZOS)

    def _run_model_pipe(self, seed: Optional[int] = None, **kwargs):
        if seed is not None:
            kwargs.update(dict(generator=torch.Generator(device=self.device).manual_seed(seed)))
        image = kwargs.pop("image", self._empty_image)
        result = self.pipe(
                image=image,
                **self.pipe_config,
                **kwargs)
        return result
    
    def _run_edit_pipe(self,
                       prompt_images: Iterable[Image.Image],
                       seed: Optional[int] = None,
                       **kwargs):
        prompt_images = list(self._prepare_input_image(prompt_image) for prompt_image in prompt_images)

        return self._run_model_pipe(seed=seed, image=prompt_images, **kwargs)
    
    
    def edit_image(self, prompt_image: Image.Image | Iterable[Image.Image], seed: int, prompting: Prompting | str):
        """ 
        Edit the image using Qwen Edit.

        Args:
            prompt_image: The prompt image to edit.
            prompting: Prompting object or string prompt.

        Returns:
            The edited image.
        """
        if self.pipe is None:
            logger.error("Edit Model is not loaded")
            raise RuntimeError("Edit Model is not loaded")
        
        try:
            start_time = time.time()

            # Convert string to TextPrompting if needed
            if isinstance(prompting, str):
                prompting = TextPrompting(positive=prompting)

            prompt_images = list(prompt_image) if isinstance(prompt_image, Iterable) else [prompt_image]

            prompting_args = prompting.model_dump()

            # Run the edit pipe
            result = self._run_edit_pipe(prompt_images=prompt_images,
                                        **prompting_args,
                                        seed=seed)
            
            generation_time = time.time() - start_time
            
            results = tuple(result.images)
            
            logger.success(f"Edited image generated in {generation_time:.2f}s, Size: {results[0].size}, Seed: {seed}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise e