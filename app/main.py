import logging
logging.basicConfig(level=logging.INFO)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.routers import dataset, webcam
from app.core.database import engine, Base
import onnxruntime as ort

logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Graceful shutdown: release the camera before the process exits
    from app.services.webcam_service import webcam_service
    if webcam_service.is_running:
        logger.info("Shutdown: stopping webcam loop …")
        webcam_service.stop()


app = FastAPI(
    title="Face Recognition API",
    description="",
    version="0.1.0",
    lifespan=lifespan,
)

try:
    providers = ort.get_available_providers()
    logger.info(f"ONNX Runtime available providers: {providers}")
except Exception as e:
    logger.warning(f"Failed to check ONNX Runtime providers: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(dataset.router)
app.include_router(webcam.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to Face Recognition API"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)