from typing import Optional
from typing_extensions import Tuple
from pydantic import BaseModel


class BackgroundRemovalConfig(BaseModel):
    """Background removal configuration"""
    model_id: str = "ZhengPeng7/BiRefNet"
    input_image_size: Tuple[int, int] = (1024, 1024)
    output_image_size: Optional[Tuple[int, int]] = None
    padding_percentage: float = 0.0
    limit_padding: bool = True
    gpu: int = 0
