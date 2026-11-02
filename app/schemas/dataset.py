from pydantic import BaseModel
from typing import List, Optional
from fastapi import UploadFile

class DatasetBase(BaseModel):
    label: str

class DatasetCreate(BaseModel):
    label: str
    embedding: List[float]

class DatasetResponse(BaseModel):
    id: int
    label: str
    
    class Config:
        from_attributes = True

class DatasetUpload(BaseModel):
    message: str
    processed_count: int
    failed_count: int
    labels: List[str]