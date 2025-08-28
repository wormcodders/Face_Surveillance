# ArcFace ONNX recognizer with CPU/GPU toggle
import os
import cv2
import numpy as np
import onnxruntime as ort

class FaceRecognizer:
    def __init__(self, model_path: str, db_dir: str, use_gpu: bool = False):
        # choose providers
        providers = ["CPUExecutionProvider"]
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("[FaceRecognizer] Using CUDAExecutionProvider")
        else:
            print("[FaceRecognizer] Using CPUExecutionProvider")

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # database (embeddings, labels)
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        emb_file = os.path.join(self.db_dir, "embeddings.npy")
        lbl_file = os.path.join(self.db_dir, "labels.npy")
        self.embeddings = np.load(emb_file) if os.path.exists(emb_file) else None
        self.labels = np.load(lbl_file) if os.path.exists(lbl_file) else None

    def preprocess(self, face):
        # face: BGR image crop
        face = cv2.resize(face, (112,112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # to (1,3,112,112)
        inp = np.transpose(face, (2,0,1))[None, ...].astype(np.float32)
        return inp

    def get_embedding(self, face):
        inp = self.preprocess(face)
        out = self.session.run(None, {self.input_name: inp})[0]
        emb = out.reshape(-1)
        emb = emb / np.linalg.norm(emb)
        return emb

    def match(self, emb, threshold: float = 0.35):
        """
        emb: normalized 1D vector
        returns (name, score) where score = cosine distance (lower better)
        """
        if self.embeddings is None or self.labels is None:
            return "Unknown", 1.0
        # embeddings stored as (N, D) normalized
        sims = np.dot(self.embeddings, emb)  # cosine similarity
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        # convert sim to distance-like measure (0 best, 1 worst)
        dist = 1.0 - best_sim
        if dist <= threshold:
            return str(self.labels[best_idx]), dist
        else:
            return "Unknown", dist

    def reload_db(self):
        emb_file = os.path.join(self.db_dir, "embeddings.npy")
        lbl_file = os.path.join(self.db_dir, "labels.npy")
        self.embeddings = np.load(emb_file) if os.path.exists(emb_file) else None
        self.labels = np.load(lbl_file) if os.path.exists(lbl_file) else None
