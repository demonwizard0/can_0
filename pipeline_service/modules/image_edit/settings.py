from pathlib import Path
from pydantic import BaseModel


class QwenConfig(BaseModel):
    """Qwen model configuration"""
    base_model_path: str = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
    model_path: str = "Qwen/Qwen-Image-Edit-2511"
    lora_path: str = "lightx2v/Qwen-Image-Edit-2511-Lightning"
    lora_angles_path: str = "" 
    lora_angles_filename: str = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 8
    true_cfg_scale: float = 1.0
    prompt_path_base: Path = Path("prompts") / "qwen_edit_prompt_v1.json"
    gpu: int = 0
    dtype: str = "bf16"
