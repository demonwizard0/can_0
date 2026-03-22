from __future__ import annotations

import base64
import io
import time
from datetime import datetime
from typing import Optional
from urllib import response

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from modules.converters.params import GLBConverterParams
import torch
import gc

from config.settings import SettingsConf
from config.prompting_library import PromptingLibrary
from logger_config import logger
from schemas.requests import GenerationRequest
from schemas.responses import GenerationResponse
from modules.mesh_generator.schemas import TrellisParams, TrellisRequest, TrellisResult
from modules.image_edit.qwen_edit_module import QwenEditModule
from modules.background_removal.ben2_module import BEN2BackgroundRemovalService
from modules.background_removal.birefnet_module import BirefNetBackgroundRemovalService
from modules.grid_renderer.render import GridViewRenderer
from modules.mesh_generator.trellis_manager import TrellisService
from modules.converters.glb_converter import GLBConverter
from libs.trellis2.representations.mesh.base import MeshWithVoxel
from modules.utils import image_grid, secure_randint, set_random_seed, decode_image, to_png_base64, save_files
    
class GenerationPipeline:
    """
    Generation pipeline 
    """

    def __init__(self, settings: SettingsConf, renderer: Optional[GridViewRenderer] = None) -> None:
        self.settings = settings
        self.renderer = renderer

        # Initialize modules
        self.qwen_edit = QwenEditModule(settings.qwen, settings.model_versions)

        # Initialize background removal module
        if self.settings.background_removal.model_id == "PramaLLC/BEN2":
            self.rmbg = BEN2BackgroundRemovalService(settings.background_removal, settings.model_versions)
        elif self.settings.background_removal.model_id == "ZhengPeng7/BiRefNet_dynamic":
            self.rmbg = BirefNetBackgroundRemovalService(settings.background_removal, settings.model_versions)
        elif self.settings.background_removal.model_id == "ZhengPeng7/BiRefNet":
            self.rmbg = BirefNetBackgroundRemovalService(settings.background_removal, settings.model_versions)
        else:
            raise ValueError(f"Unsupported background removal model: {self.settings.background_removal.model_id}")

        # Initialize prompting libraries for both modes
        self.prompting_library = PromptingLibrary.from_file(settings.qwen.prompt_path_base)

        # Initialize Trellis module
        self.trellis = TrellisService(settings.trellis, settings.model_versions)
        self.glb_converter = GLBConverter(settings.glb_converter)
        
    async def startup(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting pipeline")
        self.settings.output.output_dir.mkdir(parents=True, exist_ok=True)

        await self.qwen_edit.startup()
        await self.rmbg.startup()
        await self.trellis.startup()
        
        logger.info("Warming up generator...")
        await self.warmup_generator()
        self._clean_gpu_memory()
        
        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        logger.info("Closing pipeline")

        # Shutdown all modules
        await self.qwen_edit.shutdown()
        await self.rmbg.shutdown()
        await self.trellis.shutdown()

        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        """
        Clean the GPU memory.
        """
        gc.collect()
        torch.cuda.empty_cache()

    async def warmup_generator(self) -> None:
        """Function for warming up the generator"""
        
        temp_image = Image.new("RGB",(512,512),color=(128,128,128))
        buffer = io.BytesIO()
        temp_image.save(buffer, format="PNG")
        temp_image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(temp_image_bytes).decode("utf-8")

        request = GenerationRequest(
            prompt_image=image_base64,
            prompt_type="image",
            seed=42
        )

        result = await self.generate(request)
        
        if result.glb_file_base64 and self.renderer:
            grid_view_bytes = self.renderer.grid_from_glb_bytes(result.glb_file_base64)
            if not grid_view_bytes:
                logger.warning("Grid view generation failed during warmup")

    async def generate_from_upload(self, image_bytes: bytes, seed: int) -> bytes:
        """
        Generate 3D model from uploaded image file and return GLB as bytes.
        
        Args:
            image_bytes: Raw image bytes from uploaded file
            seed: Random seed for generation
            output_type: Desired output type (MESH) (default: MESH)
            
        Returns:
            GLB file as bytes
        """
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Create request
        request = GenerationRequest(
            prompt_image=image_base64,
            prompt_type="image",
            seed=seed
        )

        response = await self.generate(request)
        
        return response.glb_file_base64 # bytes
    
    def _get_dynamic_glb_params(self, mesh: MeshWithVoxel, request_params, elapsed_time: float):
        """
        Intelligent GLB parameter selection based on remaining time budget.
        Fast tasks get higher quality processing (texture, decimation).
        Slow tasks get reduced params to stay within timeout.
        """
        TIME_TARGET = 78  
        remaining = TIME_TARGET - elapsed_time
        face_count = mesh.faces.shape[0]

        if remaining > 55:
            dynamic = GLBConverterParams.Overrides(
                texture_size=3072,
                decimation_target=400000  
            )
            logger.debug(f"Dynamic GLB: FAST ({elapsed_time:.0f}s used, {remaining:.0f}s left, {face_count} faces) -> tex=3072, decim=400k")
        elif remaining > 40:
            dynamic = GLBConverterParams.Overrides(
                texture_size=2560,
                decimation_target=300000
            )
            logger.debug(f"Dynamic GLB: NORMAL ({elapsed_time:.0f}s used, {remaining:.0f}s left, {face_count} faces) -> tex=2560, decim=300k")
        elif remaining > 25:
            logger.debug(f"Dynamic GLB: MODERATE ({elapsed_time:.0f}s used, {remaining:.0f}s left, {face_count} faces) -> defaults")
            return request_params
        else:
            dynamic = GLBConverterParams.Overrides(
                texture_size=1536,
                decimation_target=180000
            )
            logger.debug(f"Dynamic GLB: SLOW ({elapsed_time:.0f}s used, {remaining:.0f}s left, {face_count} faces) -> tex=1536, decim=180k")

        if request_params:
            merged = dynamic.model_dump(exclude_none=True)
            merged.update(request_params.model_dump(exclude_none=True))
            return GLBConverterParams.Overrides(**merged)
        
        return dynamic

    def _preprocess_for_qwen(self, image: Image.Image) -> Image.Image:
        """
        Apply safe sharpening + contrast before sending to Qwen.
        """
    
        # --- 1. Sharpen (mild) ---
        # image = image.filter(ImageFilter.UnsharpMask(
        #     radius=1.5,     # small radius
        #     percent=120,    # strength (100~150 safe)
        #     threshold=3
        # ))
    
        # --- 2. Contrast (controlled) ---
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.3)  # 1.1 ~ 1.3 recommended
    
        # --- 3. Optional: slight color normalization ---
        # color = ImageEnhance.Color(image)
        # image = color.enhance(1.05)
    
        return image
        
    def _edit_images(self, image: Image.Image, seed: int) -> list[Image.Image]:
        """
        Edit image based on current mode (multiview or base).
        
        Args:
            image: Input image to edit
            seed: Random seed for reproducibility
            
        Returns:
            List of edited images
        """

        image = self._preprocess_for_qwen(image)
        if self.settings.trellis.multiview:
            logger.info("Multiview mode: 1 base prompt x2 seeds for multidiffusion")
            base_prompt = self.prompting_library.promptings['base']

            edited_images = []
            for i in range(2):
                run_seed = seed + i
                logger.debug(f"Editing with base prompt (seed={run_seed})")
                result = self.qwen_edit.edit_image(
                    prompt_image=image,
                    seed=run_seed,
                    prompting=base_prompt
                )
                edited_images.extend(result)

            return edited_images
        
        # Base mode: only clean background, single view (1 image)
        logger.info("Base mode: single view with background cleaning and rotation")
        base_prompt = self.prompting_library.promptings['base']
        logger.debug(f"Editing base view with prompt: {base_prompt}")
        base_prompt = (
            "Show the SAME object in a consistent three-quarters view. "
            "The object must remain identical in shape, proportions, and details. "
            "Do not change geometry, size, or structure. "
            "Do not redesign or reinterpret the object. "
            "Keep all features exactly the same across views. "
            "Preserve original color, texture, and material. "
            "Remove background clutter and use a clean plain background. "
            "Use soft, even lighting with no reflections or strong shadows."
        )
        return self.qwen_edit.edit_image(
            prompt_image=image,
            seed=seed,
            prompting=base_prompt
        )

    async def generate_mesh(self, request: GenerationRequest) -> tuple[MeshWithVoxel, list[Image.Image], list[Image.Image]]:
        """
        Generate mesh from Trellis pipeline, along with processed images.

        Args:
            request: Generation request with prompt and settings

        Returns:
            Tuple of (mesh, images_edited, images_without_background)
        """
        # Set seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
        set_random_seed(request.seed)

        # Decode input image
        image = decode_image(request.prompt_image)

        # 1. Edit the image using Qwen Edit
        # Qwen uses its own torch.Generator per call, but re-seed global state
        # before each step to ensure determinism regardless of consumed RNG ops.
        set_random_seed(request.seed)
        images_edited = list(self._edit_images(image, request.seed))
        self._clean_gpu_memory()

        # 2. Remove background
        set_random_seed(request.seed)
        images_with_background = list(image.copy() for image in images_edited)
        images_without_background = self.rmbg.remove_background(images_with_background)
        self._clean_gpu_memory()

        # Resolve Trellis parameters from request
        trellis_params: TrellisParams = request.trellis_params
       
        # 3. Generate the 3D model
        # Re-seed so Trellis starts from a known state
        set_random_seed(request.seed)
        mesh = self.trellis.generate(
            TrellisRequest(
                image=images_without_background,
                seed=request.seed,
                params=trellis_params
            )
        )

        return mesh, images_edited, images_without_background

    def convert_mesh_to_glb(self, mesh: MeshWithVoxel, glbconv_params: GLBConverterParams) -> bytes:
        """
        Convert mesh to GLB format using GLBConverter.

        Args:
            mesh: The mesh to convert
            glbconv_params: Optional override parameters for GLB conversion

        Returns:
            GLB file as bytes
        """
        start_time = time.time()
        glb_mesh = self.glb_converter.convert(mesh, params=glbconv_params)

        buffer = io.BytesIO()
        glb_mesh.export(file_obj=buffer, file_type="glb", extension_webp=False)
        buffer.seek(0)
        
        logger.info(f"GLB conversion time: {time.time() - start_time:.2f}s")
        return buffer.getvalue()

    def prepare_outputs(
        self,
        images_edited: list[Image.Image],
        images_without_background: list[Image.Image],
        glb_trellis_result: Optional[TrellisResult]
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Prepare output files: save to disk if configured and generate base64 strings if needed.

        Args:
            images_edited: List of edited images
            images_without_background: List of images with background removed
            glb_trellis_result: Generated GLB result (optional)

        Returns:
            Tuple of (image_edited_base64, image_without_background_base64)
        """
        start_time = time.time()
        # Create grid images once for both save and send operations
        image_edited_grid = image_grid(images_edited)
        image_without_background_grid = image_grid(images_without_background)

        # Save generated files if configured
        if self.settings.output.save_generated_files:
            save_files(glb_trellis_result, image_edited_grid, image_without_background_grid)

        # Convert to PNG base64 for response if configured
        image_edited_base64 = None
        image_without_background_base64 = None
        if self.settings.output.send_generated_files:
            image_edited_base64 = to_png_base64(image_edited_grid)
            image_without_background_base64 = to_png_base64(image_without_background_grid)
            
        logger.info(f"Output preparation time: {time.time() - start_time:.2f}s")

        return image_edited_base64, image_without_background_base64

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Execute full generation pipeline with output types.
        
        Args:
            request: Generation request with prompt and settings
            
        Returns:
            GenerateResponse with generated assets
        """
        t1 = time.time()
        logger.info(f"Request received | Seed: {request.seed} | Prompt Type: {request.prompt_type.value}")

        # Generate mesh and get processed images
        mesh, images_edited, images_without_background = await self.generate_mesh(request)

        glb_trellis_result = None
        
        self._clean_gpu_memory()

        # Convert mesh to GLB
        if mesh:
            elapsed = time.time() - t1
            dynamic_params = self._get_dynamic_glb_params(mesh, request.glbconv_params, elapsed) if self.settings.api.dynamic_params else request.glbconv_params
            glb_bytes = self.convert_mesh_to_glb(mesh, dynamic_params)
            glb_trellis_result = TrellisResult(file_bytes=glb_bytes)

        # Save generated files
        image_edited_base64, image_no_bg_base64 = None, None
        if self.settings.output.save_generated_files or self.settings.output.send_generated_files:
            image_edited_base64, image_no_bg_base64 = self.prepare_outputs(
                images_edited,
                images_without_background,
                glb_trellis_result
            )

        t2 = time.time()
        generation_time = t2 - t1

        logger.success(f"Generation time: {generation_time:.2f}s")

        # Clean the GPU memory
        self._clean_gpu_memory()

        response = GenerationResponse(
            generation_time=generation_time,
            glb_file_base64=glb_trellis_result.file_bytes if glb_trellis_result else None,
            image_edited_file_base64=image_edited_base64,
            image_without_background_file_base64=image_no_bg_base64
        )
        
        return response