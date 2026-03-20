from pydantic import BaseModel, Field


class CompressParams(BaseModel):
    quality: int = Field(default=80, ge=1, le=100, description="Output quality (1–100)")
    lossless: bool = Field(default=False, description="WebP only: force lossless encoding")
