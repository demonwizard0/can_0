from pydantic import BaseModel
from .enums import TrellisPipeType, TrellisMode


class TrellisConfig(BaseModel):
    """TRELLIS.2 model configuration"""
    model_id: str = "microsoft/TRELLIS.2-4B"
    pipeline_config_path: str = "libs/trellis2/pipeline.json"
    sparse_structure_steps: int = 12
    sparse_structure_cfg_strength: float = 7.5
    shape_slat_steps: int = 12
    shape_slat_cfg_strength: float = 3.0
    tex_slat_steps: int = 12
    tex_slat_cfg_strength: float = 3.0
    pipeline_type: TrellisPipeType = TrellisPipeType.MODE_1024  # '512', '1024', '1024_cascade', '1536_cascade'
    max_num_tokens: int = 49152
    mode: TrellisMode = TrellisMode.MULTIDIFFUSION
    multiview: bool = False
    gpu: int = 0
    adaptive_pipeline: bool = False
    complexity_threshold: int = 4500
    probe_cfg_strength: float = 7.5
