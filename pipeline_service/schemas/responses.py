from typing import Optional

from pydantic import BaseModel


class GenerationResponse(BaseModel):
    generation_time: float 
    glb_file_base64: Optional[str | bytes] = None
    grid_view_file_base64: Optional[str | bytes] = None
    image_edited_file_base64: Optional[str] = None
    image_without_background_file_base64: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "generation_time": 7.2,
                "glb_file_base64": "base64_encoded_glb_file",
                "grid_view_file_base64": "base64_encoded_grid_view_file",
                "image_edited_file_base64": "base64_encoded_image_edited_file",
                "image_without_background_file_base64": "base64_encoded_image_without_background_file",
            }
        }

