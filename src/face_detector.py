# YOLOv8 face detector wrapper with CPU/GPU toggle
from ultralytics import YOLO
import warnings

class FaceDetector:
    def __init__(self, model_path: str, use_gpu: bool = False, conf: float = 0.45):
        self.model_path = model_path
        self.conf = conf
        self.model = None
        try:
            device = "cuda" if use_gpu else "cpu"
            self.model = YOLO(model_path)
            # move model to device (ultralytics handles device internally)
            if use_gpu:
                try:
                    self.model.to("cuda")
                except Exception:
                    warnings.warn("Could not move YOLO to CUDA; continuing on default device.")
            print(f"[FaceDetector] Loaded {model_path} (device={device})")
        except Exception as e:
            print(f"[FaceDetector] WARNING: failed to load YOLO model: {e}")
            self.model = None

    def detect(self, frame):
        """
        frame: BGR numpy array (OpenCV)
        returns list of (x1,y1,x2,y2) boxes (int)
        """
        if self.model is None:
            return []
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        boxes = []
        try:
            r = results[0]
            if hasattr(r, "boxes") and r.boxes is not None:
                xy = r.boxes.xyxy.cpu().numpy()
                for item in xy:
                    x1, y1, x2, y2 = map(int, item[:4])
                    boxes.append((x1, y1, x2, y2))
        except Exception:
            pass
        return boxes
