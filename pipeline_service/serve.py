from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
import base64
import asyncio

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config.settings import settings
from logger_config import logger
from schemas.requests import GenerationRequest
from schemas.responses import GenerationResponse
from modules.pipeline import GenerationPipeline
from modules.grid_renderer.render import GridViewRenderer
from utils import generate_chunks

renderer = GridViewRenderer()
pipeline = GenerationPipeline(settings, renderer=renderer)



@asynccontextmanager
async def lifespan(app: FastAPI):
    await pipeline.startup()
    try:
        yield
    finally:
        await pipeline.shutdown()

app = FastAPI(
    title=settings.api.api_title,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health() -> dict[str, str]:
    """
    Check if the service is running. 

    Returns:
        dict[str, str]: Status of the service
    """
    return {"status": "ready"}

@app.post("/generate_from_base64", response_model=GenerationResponse)
async def generate_from_base64(request: GenerationRequest) -> GenerationResponse:
    """
    Generate 3D model from base64 encoded image (JSON request).

    Returns JSON with generation_time and base64 encoded outputs.
    """
    try:
        result = await asyncio.wait_for(pipeline.generate(request), timeout=settings.api.timeout)

        if request.render_grid_view and result.glb_file_base64:
            grid_view_bytes = renderer.grid_from_glb_bytes(result.glb_file_base64)
            if grid_view_bytes is not None:
                result.grid_view_file_base64 = base64.b64encode(grid_view_bytes).decode("utf-8")

        if result.glb_file_base64:
            result.glb_file_base64 = base64.b64encode(result.glb_file_base64).decode("utf-8") # return bytes

        return result

    except asyncio.TimeoutError:
        logger.error(f"Generation timed out after {settings.api.timeout} seconds")
        raise HTTPException(status_code=408, detail="timeout") from None
    except Exception as exc:
        logger.exception(f"Error generating task: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate")
async def generate(prompt_image_file: UploadFile = File(...), seed: int = Form(-1)) -> StreamingResponse:
    """
    Upload image file and generate 3D model as GLB buffer.
    Returns binary GLB file directly.
    """
    try:
        logger.info(f"Task received. Uploading image: {prompt_image_file.filename}")

        # Generate GLB from uploaded file
        glb_bytes = await asyncio.wait_for(
            pipeline.generate_from_upload(await prompt_image_file.read(), seed),
            timeout=settings.api.timeout
        )
                                
        # Wrap bytes in BytesIO for streaming
        glb_buffer = BytesIO(glb_bytes)
        buffer_size = len(glb_buffer.getvalue())
        glb_buffer.seek(0)
        logger.info(f"Task completed. GLB size: {buffer_size} bytes")        
     
        return StreamingResponse(
            generate_chunks(glb_buffer),
            media_type="application/octet-stream",
            headers={"Content-Length": str(buffer_size)}
        )

    except asyncio.TimeoutError:
        logger.error(f"Generation timed out after {settings.api.timeout} seconds")
        raise HTTPException(status_code=408, detail="timeout") from None
    except Exception as exc:
        logger.exception(f"Error generating from upload: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/setup/info")
async def get_setup_info() -> dict:
    """
    Get current pipeline configuration for experiment logging.
    
    Returns:
        dict: Pipeline configuration settings
    """
    try:
        return settings.model_dump()
    except Exception as e:
        logger.error(f"Failed to get setup info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve configuration")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
    )
