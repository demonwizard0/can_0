from __future__ import annotations

import os
import time
from typing import Iterable, Optional

import numpy as np
from libs.trellis2.representations.mesh.base import MeshWithVoxel
import torch
from PIL import Image

from .settings import TrellisConfig
from .enums import TrellisPipeType
from config.settings import ModelVersionsConfig
from logger_config import logger
from libs.trellis2.pipelines import Trellis2ImageTo3DPipeline
from .schemas import TrellisRequest, TrellisParams

class TrellisService:
    def __init__(self, trellis_config: TrellisConfig, model_versions: ModelVersionsConfig):
        self.settings = trellis_config
        self.model_versions = model_versions
        self.pipeline: Optional[Trellis2ImageTo3DPipeline] = None
        self.gpu = trellis_config.gpu
        self.default_params = TrellisParams.from_settings(self.settings)

    def _get_model_revisions(self) -> tuple[Optional[str], dict[str, str]]:
        """
        Get pinned revisions for Trellis pipeline and related models.
        
        Returns:
            Tuple of (trellis_revision, model_revisions_dict)
        """
        trellis_revision = self.model_versions.get_revision(self.settings.model_id)
        
        model_revisions = {
            model_id: revision 
            for model_id, revision in self.model_versions.models.items()
            if model_id != self.settings.model_id
        }
        
        return trellis_revision, model_revisions

    async def startup(self) -> None:
        logger.info("Loading TRELLIS.2 pipeline...")
        os.environ.setdefault("ATTN_BACKEND", "flash-attn")
        os.environ.setdefault("SPCONV_ALGO", "native")
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu)

        trellis_revision, model_revisions = self._get_model_revisions()

        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
            self.settings.model_id,
            config_file=self.settings.pipeline_config_path,
            revision=trellis_revision,
            model_revisions=model_revisions,
        )
        
        self.pipeline.cuda()
        logger.success(f"{self.settings.model_id} pipeline ready.")

    async def shutdown(self) -> None:
        self.pipeline = None
        torch.cuda.empty_cache()
        logger.info(f"{self.settings.model_id} pipeline closed.")

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def _probe_complexity(self, images: list[Image.Image], seed: int) -> int:
        """
        Sample sparse structure with probe guidance strength and return the token count.

        A high token count indicates a geometrically complex object (many fine details,
        thin branches, etc.) that may be better served by the 512 pipeline.
        """
        generator = torch.Generator()
        generator.manual_seed(seed)
        torch.manual_seed(seed)

        # Use only the first image: sample_sparse_structure is called directly
        # (no inject_sampler_multi_image), so cond batch size must be 1.
        cond = self.pipeline.get_cond([images[0]], 512)

        coords = self.pipeline.sample_sparse_structure(
            cond,
            resolution=32,  # coarse resolution used by all cascade pipelines
            num_samples=1,
            sampler_params={
                "steps": self.settings.sparse_structure_steps,
                "guidance_strength": self.settings.probe_cfg_strength,
            },
            generator=generator,
        )
        return int(coords.shape[0])

    def _select_pipeline_type(self, images: list[Image.Image], seed: int, params: TrellisParams) -> TrellisParams:
        """
        Run the complexity probe and downgrade pipeline_type to 512 if the object
        has too many sparse tokens, indicating high geometric complexity.
        Returns (possibly modified) params.
        """
        if not self.settings.adaptive_pipeline:
            return params

        # Only probe when the configured pipeline is higher than 512
        if params.pipeline_type == TrellisPipeType.MODE_512:
            return params

        probe_start = time.time()
        num_tokens = self._probe_complexity(images, seed)
        probe_elapsed = time.time() - probe_start

        if num_tokens > self.settings.complexity_threshold:
            logger.info(
                f"Complexity probe: {num_tokens} tokens (>{self.settings.complexity_threshold}) "
                f"in {probe_elapsed:.2f}s → switching to 512 pipeline for coherence"
            )
            return params.model_copy(update={"pipeline_type": TrellisPipeType.MODE_512})

        logger.info(
            f"Complexity probe: {num_tokens} tokens (<={self.settings.complexity_threshold}) "
            f"in {probe_elapsed:.2f}s → keeping {params.pipeline_type.value} pipeline"
        )
        return params

    def generate(
        self,
        request: TrellisRequest,
    ) -> MeshWithVoxel:
        if not self.pipeline:
            raise RuntimeError(f"{self.settings.model_id} pipeline not loaded.")

        images = request.image if isinstance(request.image, Iterable) else [request.image]
        images_rgb = [image.convert("RGB") for image in images]
        num_images = len(images_rgb)
        
        params = self.default_params.overrided(request.params)
        params = self._select_pipeline_type(images_rgb, request.seed, params)
        
        logger.info(f"Generating Trellis {request.seed=} and image size {images[0].size} (Using {num_images} images) | Pipeline: {params.pipeline_type.value} | Max Tokens: {params.max_num_tokens} | {'Mode: ' + params.mode.value if params.mode.value else ''}")
        
        logger.debug(f"Trellis generation parameters: {params}")

        start = time.time()
        try:
            # Run TRELLIS.2 pipeline - returns list of MeshWithVoxel
            meshes = self.pipeline.run_multi_image(
                images=images_rgb,
                seed=request.seed,
                sparse_structure_sampler_params={
                    "steps": params.sparse_structure_steps,
                    "guidance_strength": params.sparse_structure_cfg_strength,
                },
                shape_slat_sampler_params={
                    "steps": params.shape_slat_steps,
                    "guidance_strength": params.shape_slat_cfg_strength,
                },
                tex_slat_sampler_params={
                    "steps": params.tex_slat_steps,
                    "guidance_strength": params.tex_slat_cfg_strength,
                },
                mode=params.mode,
                pipeline_type=params.pipeline_type,
                max_num_tokens=params.max_num_tokens,
            )

            mesh = meshes[0]
            mesh.simplify()

            generation_time = time.time() - start
            logger.info(f"{self.settings.model_id} mesh generated in {generation_time:.2f}s")

            total_time = time.time() - start
            logger.success(
                f"{self.settings.model_id} finished in {total_time:.2f}s. "
            )
            return mesh

        finally:
            torch.cuda.empty_cache()
