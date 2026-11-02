import os
import tempfile
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.dataset import DatasetCreate, DatasetResponse, DatasetUpload
from app.services.dataset_service import  dataset_service

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.post("/upload", response_model=DatasetUpload, status_code=status.HTTP_201_CREATED)
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Endpoint to upload a ZIP file containing labeled face images."""
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only ZIP files are allowed.")

    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name

        zip_name = os.path.splitext(file.filename)[0]
        result = dataset_service.process_zip_file(tmp_file_path, db, zip_name=zip_name)

        return DatasetUpload(
            message=(
                f"Dataset processed: {result['processed_count']} saved"
                + (f", {result['failed_count']} skipped (no face detected)." if result['failed_count'] else ".")
            ),
            processed_count=result["processed_count"],
            failed_count=result["failed_count"],
            labels=result["labels"],
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

@router.get("/", response_model=List[DatasetResponse])
def get_datasets(db: Session = Depends(get_db)):
    """Endpoint to retrieve all datasets."""
    try:
        return dataset_service.get_all_datasets(db)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Endpoint to retrieve a dataset by its ID."""
    try:
        dataset = dataset_service.get_dataset_by_id(dataset_id, db)
        if dataset is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")
        return dataset
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    

@router.get("/label/{label}", response_model=List[DatasetResponse])
def get_datasets_by_label(label: str, db: Session = Depends(get_db)):
    """Endpoint to retrieve datasets by label."""
    try:
        datasets = dataset_service.get_datasets_by_label(label, db)
        if not datasets:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No datasets found with the specified label.")
        return datasets
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
@router.delete("/label/{label}", response_model=dict)
def delete_datasets_by_label(label: str, db: Session = Depends(get_db)):
    """Endpoint to delete datasets by label."""
    try:
        deleted_count = dataset_service.delete_datasets(label, db)
        if deleted_count == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No datasets found with the specified label.")
        return {"message": f"Deleted {deleted_count} dataset(s) with label '{label}'."}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))