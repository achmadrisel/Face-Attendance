import logging
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from boxmot.trackers.botsort.botsort import BotSort
import cv2
import numpy as np
import onnxruntime
from insightface.model_zoo import arcface_onnx, scrfd
from insightface.utils import face_align
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dataset import Dataset

logger = logging.getLogger(__name__)

# ── Confidence thresholds (cosine similarity, glintr100) ─────────────────────
_HIGH_THRESH = 0.75   # green  – confident match
_MID_THRESH  = 0.60   # yellow – possible match
# below _MID_THRESH  → red / unknown

_COLOR: Dict[str, Tuple[int, int, int]] = {
    "high": (0, 255, 0),    # green  (BGR)
    "mid":  (0, 255, 255),  # yellow
    "low":  (0, 0, 255),    # red
}


# ── RAM embedding cache ───────────────────────────────────────────────────────
@dataclass
class EmbeddingCache:
    labels: List[str]      # (N,)
    matrix: np.ndarray     # (N, 512) – L2-normalised float32 rows

    @property
    def size(self) -> int:
        return len(self.labels)


def _build_cache(db: Session) -> EmbeddingCache:
    rows = db.query(Dataset.label, Dataset.embedding).all()
    if not rows:
        return EmbeddingCache(labels=[], matrix=np.empty((0, 512), dtype=np.float32))
    labels = [r.label for r in rows]
    matrix = np.array([r.embedding for r in rows], dtype=np.float32)
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / (norms + 1e-8)
    return EmbeddingCache(labels=labels, matrix=matrix)


# ── WebcamService ─────────────────────────────────────────────────────────────
class WebcamService:
    def __init__(self):
        # ── ONNX Runtime provider selection ──────────────────────────────────
        try:
            onnxruntime.preload_dlls(cuda=True, cudnn=True)
        except AttributeError:
            pass
        except Exception as e:
            logger.error(f"Error preloading ONNX Runtime DLLs: {e}")

        available = onnxruntime.get_available_providers()
        logger.info(f"Available ONNX Runtime providers: {available}")

        if "CUDAExecutionProvider" in available:
            self._providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._ctx       = 0
        elif "OpenVINOExecutionProvider" in available:
            self._providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            self._ctx       = 0
        elif "ROCMExecutionProvider" in available:
            self._providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
            self._ctx       = 0
        else:
            self._providers = ["CPUExecutionProvider"]
            self._ctx       = -1

        logger.info(f"Using providers: {self._providers}")

        # ── Load face models ──────────────────────────────────────────────────
        def _make_so() -> onnxruntime.SessionOptions:
            so = onnxruntime.SessionOptions()
            so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            return so

        try:
            det_sess = onnxruntime.InferenceSession(
                settings.FACE_DETECTION, sess_options=_make_so(), providers=self._providers
            )
            self.face_detector = scrfd.SCRFD(model_file=settings.FACE_DETECTION)
            self.face_detector.session = det_sess
            self.face_detector.prepare(ctx_id=self._ctx, input_size=(640, 640))
            logger.info(f"Detector  active provider: {det_sess.get_providers()[0]}")

            rec_sess = onnxruntime.InferenceSession(
                settings.FACE_RECOGNITION, sess_options=_make_so(), providers=self._providers
            )
            self.face_recognizer = arcface_onnx.ArcFaceONNX(model_file=settings.FACE_RECOGNITION)
            self.face_recognizer.session = rec_sess
            self.face_recognizer.prepare(ctx_id=self._ctx)
            logger.info(f"Recognizer active provider: {rec_sess.get_providers()[0]}")

            logger.info("Face detection and recognition models loaded.")
        except Exception as e:
            logger.error(f"Failed to load ONNX models: {e}")
            raise RuntimeError(f"Failed to load ONNX models: {e}")

        # ── Thread state ──────────────────────────────────────────────────────
        self._thread:       Optional[threading.Thread] = None
        self._stop_event:   Optional[threading.Event]  = None
        self._camera_index: Optional[int]              = None
        self._cache:        Optional[EmbeddingCache]   = None
        self._cache_lock:   threading.Lock             = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, camera_index: int, db: Session, frame_skip: int = 5) -> dict:
        if self.is_running:
            raise RuntimeError("Webcam is already running. Call /webcam/stop first.")

        logger.info("Loading embedding cache from database …")
        with self._cache_lock:
            self._cache = _build_cache(db)
        logger.info(f"Cache loaded: {self._cache.size} face embeddings.")

        self._stop_event   = threading.Event()
        self._camera_index = camera_index
        self._thread       = threading.Thread(
            target=self._run_loop, args=(camera_index, frame_skip), daemon=True
        )
        self._thread.start()
        logger.info(f"Webcam started on camera {camera_index} (frame_skip={frame_skip}).")
        return {
            "is_running":   True,
            "message":      f"Webcam started on camera {camera_index} with {self._cache.size} faces.",
            "camera_index": camera_index,
        }

    def stop(self) -> dict:
        if not self.is_running:
            return {"is_running": False, "message": "Webcam is not running.", "camera_index": None}
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        time.sleep(0.5)   # give OS time to fully release the camera device
        self._thread       = None
        self._stop_event   = None
        self._camera_index = None
        logger.info("Webcam loop stopped.")
        return {"is_running": False, "message": "Webcam stopped.", "camera_index": None}

    def status(self) -> dict:
        if self.is_running:
            return {"is_running": True, "message": "Webcam is running.", "camera_index": self._camera_index}
        return {"is_running": False, "message": "Webcam is not running.", "camera_index": None}

    def reload_cache(self, db: Session) -> int:
        """Atomically replace the in-RAM embedding cache. Safe to call while running."""
        new_cache = _build_cache(db)
        with self._cache_lock:
            self._cache = new_cache
        logger.info(f"Embedding cache reloaded: {new_cache.size} faces.")
        return new_cache.size

    # ── Similarity search (pure NumPy, no DB round-trip) ─────────────────────

    def _similarity_search(self, embedding: np.ndarray) -> Tuple[str, float, str]:
        with self._cache_lock:
            cache = self._cache

        if cache is None or cache.size == 0:
            return "Unknown", 0.0, "low"

        sims  = embedding @ cache.matrix.T   # (N,) cosine similarities
        best  = int(sims.argmax())
        sim   = float(sims[best])
        label = cache.labels[best]

        if sim >= _HIGH_THRESH:
            level = "high"
        elif sim >= _MID_THRESH:
            level = "mid"
        else:
            level, label = "low", "Unknown"

        return label, sim, level

    # ── Main loop — single thread, detect+track+recognize every frame_skip frames ─

    def _run_loop(self, camera_index: int, frame_skip: int) -> None:
        tracker = BotSort(
            reid_model=None,
            with_reid=False,
            cmc_method=None,
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30,
            min_hits=1,
        )

        track_labels: Dict[int, Tuple[str, float, str]] = {}
        last_tracks = np.empty((0, 8), dtype=np.float32)
        kpss: Optional[np.ndarray] = None
        frame_idx = 0
        frame_times: deque = deque(maxlen=30)
        fps = 0.0

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error(f"Cannot open camera {camera_index}.")
            self._stop_event.set()
            return

        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        cv2.startWindowThread()
        logger.info("Webcam loop started.")

        try:
            while not self._stop_event.is_set():
                t_start = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame – camera disconnected?")
                    break
                frame = cv2.flip(frame, 1)

                if frame_idx % frame_skip == 0:
                    # ── Detection ────────────────────────────────────────────────
                    bboxes, kpss = self.face_detector.detect(frame, max_num=0)

                    if bboxes is not None and bboxes.shape[0] > 0:
                        dets = np.hstack([bboxes, np.zeros((len(bboxes), 1), dtype=np.float32)])
                    else:
                        dets = np.empty((0, 6), dtype=np.float32)

                    # ── BotSort update ────────────────────────────────────────────
                    tracks = np.array(tracker.update(dets, frame))

                    if tracks.shape[0] > 0:
                        last_tracks = tracks
                    else:
                        tracks = last_tracks

                    # ── Recognition ───────────────────────────────────────────────
                    active_tids: set = set()

                    if tracks.shape[0] > 0:
                        for t in tracks:
                            x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                            tid             = int(t[4])
                            det_idx         = int(t[7]) if t.shape[0] > 7 else -1
                            active_tids.add(tid)

                            if det_idx >= 0 and kpss is not None and det_idx < len(kpss):
                                face = face_align.norm_crop(frame, landmark=kpss[det_idx])
                            else:
                                crop = frame[max(0, y1):y2, max(0, x1):x2]
                                if crop.size == 0:
                                    continue
                                face = cv2.resize(crop, (112, 112))

                            emb = self.face_recognizer.get_feat(face).flatten()
                            emb = emb / (np.linalg.norm(emb) + 1e-8)
                            track_labels[tid] = self._similarity_search(emb)

                        for tid in list(track_labels.keys()):
                            if tid not in active_tids:
                                del track_labels[tid]
                else:
                    tracks = last_tracks

                frame_idx += 1

                # ── Draw boxes + labels ───────────────────────────────────────────
                for t in tracks:
                    x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                    tid             = int(t[4])
                    label, sim, level = track_labels.get(tid, ("?", 0.0, "low"))
                    color = _COLOR[level]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{label} {sim:.2f}",
                        (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2,
                    )

                # ── FPS overlay ───────────────────────────────────────────────────
                frame_times.append(time.perf_counter() - t_start)
                if len(frame_times) == frame_times.maxlen:
                    fps = len(frame_times) / sum(frame_times)
                cv2.putText(
                    frame, f"FPS: {fps:.1f}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )

                cv2.imshow("Face Attendance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception as e:
            logger.error(f"Webcam loop error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam loop exited.")


webcam_service = WebcamService()
