# register new identities: saves embeddings.npy and labels.npy in database directory
import os
import numpy as np

class FaceRegister:
    def __init__(self, db_dir: str, recognizer: 'FaceRecognizer'):
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        self.recognizer = recognizer  # used to compute embeddings via get_embedding

    def add_images_for_person(self, person_name: str, image_files: list):
        """
        image_files: list of bytes-like objects (file.read() results), or file paths
        returns: (success_count, total)
        """
        person_dir = os.path.join(self.db_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        saved = 0
        for i, f in enumerate(image_files):
            # f can be File-like (gradio) or path string
            if isinstance(f, (bytes, bytearray)):
                path = os.path.join(person_dir, f"{person_name}_{i}.jpg")
                with open(path, "wb") as out:
                    out.write(f)
                saved += 1
            elif isinstance(f, str):
                # file path already on disk
                import shutil
                dst = os.path.join(person_dir, os.path.basename(f))
                shutil.copy(f, dst)
                saved += 1
            else:
                # assume file-like with .read()
                try:
                    content = f.read()
                    path = os.path.join(person_dir, f"{person_name}_{i}.jpg")
                    with open(path, "wb") as out:
                        out.write(content)
                    saved += 1
                except Exception:
                    pass
        # After saving images, compute embeddings per person and append to DB
        self._compute_and_append_embeddings(person_name)
        return saved, len(image_files)

    def _compute_and_append_embeddings(self, person_name: str):
        """
        For the newly added person, compute embeddings for all images in person dir,
        take mean embedding (normalized) and append to embeddings.npy and labels.npy.
        """
        person_dir = os.path.join(self.db_dir, person_name)
        files = [os.path.join(person_dir, p) for p in os.listdir(person_dir) if p.lower().endswith(('.jpg','.jpeg','.png'))]
        embs = []
        import cv2
        for p in files:
            img = cv2.imread(p)
            if img is None:
                continue
            e = self.recognizer.get_embedding(img)
            embs.append(e)
        if not embs:
            return
        mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        emb_file = os.path.join(self.db_dir, "embeddings.npy")
        lbl_file = os.path.join(self.db_dir, "labels.npy")
        if os.path.exists(emb_file) and os.path.exists(lbl_file):
            old_embs = np.load(emb_file)
            old_lbls = np.load(lbl_file)
            new_embs = np.vstack([old_embs, mean_emb.reshape(1, -1)])
            new_lbls = np.append(old_lbls, person_name)
        else:
            new_embs = mean_emb.reshape(1, -1)
            new_lbls = np.array([person_name], dtype=object)
        np.save(emb_file, new_embs)
        np.save(lbl_file, new_lbls)
        # reload recognizer db
        self.recognizer.reload_db()
