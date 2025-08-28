# Camera workers: each camera runs in its own thread and stores latest frame (BGR numpy)
import threading
import time
import cv2

class CameraWorker(threading.Thread):
    def __init__(self, cam_id, source, fps_limit=15):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.source = source
        self.fps_limit = fps_limit
        self.cap = None
        self.latest_frame = None
        self.running = threading.Event()
        self.running.clear()

    def run(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            # could not open
            return
        self.running.set()
        last_time = 0
        while self.running.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.2)
                continue
            now = time.time()
            if now - last_time < 1.0 / self.fps_limit:
                time.sleep(max(0, 1.0/self.fps_limit - (now - last_time)))
            last_time = time.time()
            self.latest_frame = frame
        try:
            self.cap.release()
        except Exception:
            pass

    def stop(self):
        self.running.clear()
        self.join(timeout=1.0)

class CameraManager:
    def __init__(self):
        self.workers = {}  # cam_id -> CameraWorker

    def add_camera(self, cam_id, source):
        if cam_id in self.workers:
            self.remove_camera(cam_id)
        w = CameraWorker(cam_id, source)
        self.workers[cam_id] = w
        w.start()

    def remove_camera(self, cam_id):
        if cam_id in self.workers:
            w = self.workers.pop(cam_id)
            w.stop()

    def list_cameras(self):
        return list(self.workers.keys())

    def get_frame(self, cam_id):
        w = self.workers.get(cam_id)
        if not w:
            return None
        return w.latest_frame
