from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.webcam import WebcamStartRequest, WebcamStatusResponse
from app.services.webcam_service import webcam_service

router = APIRouter(prefix="/webcam", tags=["webcam"])


@router.post("/start", response_model=WebcamStatusResponse)
def start_webcam(request: WebcamStartRequest, db: Session = Depends(get_db)):
    """Open the webcam, load the embedding cache from DB, and start tracking."""
    try:
        result = webcam_service.start(request.camera_index, db, request.frame_skip)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/stop", response_model=WebcamStatusResponse)
def stop_webcam():
    """Signal the webcam loop to stop and release the camera."""
    try:
        return webcam_service.stop()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/status", response_model=WebcamStatusResponse)
def webcam_status():
    """Return whether the webcam loop is currently running."""
    return webcam_service.status()


@router.post("/reload", response_model=dict)
def reload_embeddings(db: Session = Depends(get_db)):
    """Reload the face embedding cache from DB without restarting the webcam.

    Call this after uploading new faces so the running loop picks them up.
    """
    try:
        count = webcam_service.reload_cache(db)
        return {"message": f"Embedding cache reloaded: {count} faces loaded."}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
