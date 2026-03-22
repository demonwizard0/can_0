from io import BytesIO
from typing import AsyncGenerator

async def generate_chunks(buffer: BytesIO)->AsyncGenerator[bytes, None]:
    chunk_size = 1024 * 1024  # 1 MB
    while chunk := buffer.read(chunk_size):
        yield chunk