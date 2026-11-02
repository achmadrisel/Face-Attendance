import os
import zipfile
import tempfile
from typing import List, Optional
import logging
import numpy as np
import cv2
import onnxruntime
from sqlalchemy.orm import Session
from sqlalchemy import delete
from insightface.model_zoo import scrfd, arcface_onnx
from insightface.utils import face_align
from app.core.config import settings
from app.core.database import get_db
from app.models.dataset import Dataset
from app.schemas.dataset import DatasetCreate, DatasetResponse

logger = logging.getLogger(__name__)

class DatasetService:
    def __init__(self):
        try:
            onnxruntime.preload_dlls(cuda=True, cudnn=True)
            logger.info("DLLs preloaded: CUDA & cuDNN")
        except AttributeError:
            logger.warning("onnxruntime.preload_dlls() not available. Upgrade onnxruntime-gpu.")
        except Exception as e:
            logger.error(f"Error preloading ONNX Runtime DLLs: {e}")

        # Check available providers
        available_providers = onnxruntime.get_available_providers()
        logger.info(f"Available ONNX Runtime providers: {available_providers}")

        # Choose providers dynamically (preference: CUDA > OpenVINO > CPU)
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx = 0
            logger.info("Using CUDAExecutionProvider for ONNX Runtime.")
        elif "OpenVINOExecutionProvider" in available_providers:
            providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            ctx = 0   # must be >=0: insightface.prepare(ctx_id<0) forces CPUExecutionProvider
            logger.info("Using OpenVINOExecutionProvider for ONNX Runtime.")
        else:
            providers = ["CPUExecutionProvider"]
            ctx = -1
            logger.info("Using CPUExecutionProvider for ONNX Runtime.")

        def _make_so() -> onnxruntime.SessionOptions:
            so = onnxruntime.SessionOptions()
            so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            return so

        det_sess = onnxruntime.InferenceSession(
            settings.FACE_DETECTION, sess_options=_make_so(), providers=providers
        )
        self.face_detector = scrfd.SCRFD(model_file=settings.FACE_DETECTION)
        self.face_detector.session = det_sess
        self.face_detector.prepare(ctx_id=ctx, input_size=(640, 640))

        rec_sess = onnxruntime.InferenceSession(
            settings.FACE_RECOGNITION, sess_options=_make_so(), providers=providers
        )
        self.face_recognizer = arcface_onnx.ArcFaceONNX(model_file=settings.FACE_RECOGNITION)
        self.face_recognizer.session = rec_sess
        self.face_recognizer.prepare(ctx_id=ctx)
        logger.info("Dataset service models loaded.")
    
    def extract_face_embedding(self, image_path: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Extract face embedding and aligned crop from an image file.

        Returns (embedding, aligned_face) or None on failure.
        Detection is attempted first. If no face is found (e.g. the image is
        already a tight face crop), the whole image is used directly.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Could not read image: {image_path}")
                return None

            # Step 1: Detect + align (same pipeline as webcam_service)
            bboxes, kpss = self.face_detector.detect(img, max_num=1)
            if bboxes.shape[0] == 0:
                logger.warning(f"No face detected in {image_path} — skipping.")
                return None
            aligned_face = face_align.norm_crop(img, landmark=kpss[0])

            # Step 2: Extract embedding
            embedding = self.face_recognizer.get_feat(aligned_face)
            embedding = embedding.flatten()
            embedding = embedding / np.linalg.norm(embedding)

            if embedding.shape[0] != 512:
                logger.error(f"Unexpected embedding size {embedding.shape[0]}: {image_path}")
                return None

            return embedding, aligned_face
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
        
    def process_zip_file(self, zip_path: str, db: Session, zip_name: str = None) -> dict:
        """Process a ZIP file containing face images.

        Returns a dict with keys:
          processed_count – number of embeddings successfully saved
          failed_count    – number of images skipped (no face detected / read error)
          labels          – unique label names found in the ZIP
        """
        processed_count = 0
        failed_count = 0

        if not zip_name:
            zip_name = os.path.splitext(os.path.basename(zip_path))[0]

        # facedata/<zip_name>/ is the destination; zip_name is also the label
        dest_folder = os.path.join(settings.DATASET_PATH, zip_name)
        os.makedirs(dest_folder, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Collect all images recursively, regardless of internal zip structure
                img_extensions = {'.png', '.jpg', '.jpeg'}
                all_images = [
                    p for p in __import__('pathlib').Path(temp_dir).rglob('*')
                    if p.suffix.lower() in img_extensions
                ]

                for img_path in sorted(all_images):
                    result = self.extract_face_embedding(str(img_path))

                    if result is None:
                        failed_count += 1
                        continue

                    embedding, aligned_face = result
                    saved_img_path = os.path.join(dest_folder, img_path.name)
                    cv2.imwrite(saved_img_path, aligned_face)

                    db.add(Dataset(
                        label=zip_name,
                        embedding=embedding.tolist(),
                        path=saved_img_path,
                    ))
                    processed_count += 1
                    logger.info(f"Processed: {img_path.name} → {saved_img_path}")

                if processed_count == 0:
                    raise ValueError(
                        f"No faces could be extracted from the uploaded ZIP "
                        f"({failed_count} image(s) failed)."
                    )

                db.commit()
                logger.info(
                    f"ZIP processed: {processed_count} saved, "
                    f"{failed_count} failed, label={zip_name}"
                )
                return {
                    "processed_count": processed_count,
                    "failed_count": failed_count,
                    "labels": [zip_name],
                }

            except Exception as e:
                db.rollback()
                logger.error(f"Error processing ZIP {zip_path}: {e}")
                raise
            
    def get_all_datasets(self, db: Session) -> List[DatasetResponse]:
        """Retrieve all datasets from the database."""
        try:
            datasets = db.query(Dataset).all()
            return [DatasetResponse.from_orm(dataset) for dataset in datasets]
        except Exception as e:
            logger.error(f"Error retrieving datasets: {e}")
            raise Exception(f"Failed to retrieve datasets: {e}")
        
    def get_dataset_by_id(self, dataset_id: int, db: Session) -> Optional[DatasetResponse]:
        """Retrieve a dataset by its ID."""
        try:
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset is None:
                logger.warning(f"Dataset with ID {dataset_id} not found.")
                return None
            return DatasetResponse.from_orm(dataset)
        except Exception as e:
            logger.error(f"Error retrieving dataset with ID {dataset_id}: {e}")
            raise Exception(f"Failed to retrieve dataset: {e}")
    
    def get_datasets_by_label(self, label: str, db: Session) -> List[DatasetResponse]:
        """Retrieve datasets by label."""
        try:
            datasets = db.query(Dataset).filter(Dataset.label == label).all()
            if not datasets:
                logger.warning(f"No datasets found with label: {label}")
                return []
            return [DatasetResponse.from_orm(dataset) for dataset in datasets]
        except Exception as e:
            logger.error(f"Error retrieving datasets with label {label}: {e}")
            raise Exception(f"Failed to retrieve datasets: {e}")
        
    def delete_datasets(self, label: str, db: Session) -> int:
        """Delete datasets by label and return the number of deleted records."""
        try:
            delete_stmt = delete(Dataset).where(Dataset.label == label)
            result = db.execute(delete_stmt)
            db.commit()
            deleted_count = result.rowcount
            logger.info(f"Deleted {deleted_count} datasets with label: {label}")
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting datasets with label {label}: {e}")
            db.rollback()
            raise Exception(f"Failed to delete datasets: {e}")
    
dataset_service = DatasetService()