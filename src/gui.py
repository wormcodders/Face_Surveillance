# Gradio GUI: camera add/remove, switch feed, live streaming generator, registration, logs
import gradio as gr
import cv2
import numpy as np
import base64
import time
from .camera_manager import CameraManager
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer
from .face_register import FaceRegister
from .logger import PresenceLogger

class GUI:
    def __init__(self, model_face, model_rec, db_dir, use_gpu=False):
        self.detector = FaceDetector(model_face, use_gpu=use_gpu)
        self.recognizer = FaceRecognizer(model_rec, db_dir, use_gpu=use_gpu)
        self.registrar = FaceRegister(db_dir, self.recognizer)
        self.cam_mgr = CameraManager()
        self.logger = PresenceLogger()
        self.current_cam = None
        self.streaming_flags = {}  # cam_id -> Event bool

    def add_camera(self, cam_id, source):
        try:
            # If source looks like an int string, convert to int
            s = int(source) if isinstance(source, str) and source.isdigit() else source
        except Exception:
            s = source
        self.cam_mgr.add_camera(cam_id, s)
        if not self.current_cam:
            self.current_cam = cam_id
        return f"Added {cam_id}"

    def remove_camera(self, cam_id):
        self.cam_mgr.remove_camera(cam_id)
        if self.current_cam == cam_id:
            cams = self.cam_mgr.list_cameras()
            self.current_cam = cams[0] if cams else None
        return f"Removed {cam_id}"

    def list_cameras(self):
        return self.cam_mgr.list_cameras()

    def stream_generator(self, cam_id):
        # generator yielding numpy images for Gradio to stream
        self.current_cam = cam_id
        keep_streaming = True
        self.streaming_flags[cam_id] = True
        try:
            while self.streaming_flags.get(cam_id, False):
                frame = self.cam_mgr.get_frame(cam_id)
                if frame is None:
                    # yield a placeholder image or blank
                    blank = np.zeros((480,640,3), dtype=np.uint8)
                    yield blank
                    time.sleep(0.2)
                    continue

                boxes = self.detector.detect(frame)
                # do recognition for each box
                for (x1,y1,x2,y2) in boxes:
                    # clip coords
                    h, w = frame.shape[:2]
                    x1c, y1c, x2c, y2c = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
                    if x2c<=x1c or y2c<=y1c:
                        continue
                    face = frame[y1c:y2c, x1c:x2c]
                    name, dist = self.recognizer.match(self.recognizer.get_embedding(face))
                    # log presence (every frame; logger dedupes by sessions)
                    self.logger.update(name, cam_id)
                    # draw
                    color = (0,200,0) if name!="Unknown" else (0,120,255)
                    cv2.rectangle(frame, (x1c,y1c), (x2c,y2c), color, 2)
                    label = f"{name} ({dist:.2f})" if isinstance(dist, float) else str(name)
                    cv2.putText(frame, label, (x1c, max(12, y1c-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # convert BGR->RGB for Gradio
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield rgb
                time.sleep(0.03)
        finally:
            self.streaming_flags[cam_id] = False

    def stop_stream(self, cam_id):
        self.streaming_flags[cam_id] = False
        return f"Stopped {cam_id}"

    def register_person(self, person_name, files):
        # files: list of uploaded files from Gradio; we read bytes
        blobs = []
        for f in files:
            if hasattr(f, "read"):
                blobs.append(f.read())
            elif isinstance(f, bytes):
                blobs.append(f)
        saved, total = self.registrar.add_images_for_person(person_name, blobs)
        return f"Saved {saved}/{total} images for {person_name}"

    def export_logs(self):
        return self.logger.export_csv()

    def launch(self, host="0.0.0.0", port=7860, share=False):
        with gr.Blocks() as demo:
            gr.Markdown("# Face Surveillance System (YOLOv8 + ArcFace)")

            with gr.Row():
                with gr.Column(scale=1):
                    cam_id = gr.Textbox(label="Camera ID (string)")
                    cam_src = gr.Textbox(label="Camera source (rtsp url or device index as string)")
                    add_btn = gr.Button("Add Camera")
                    del_id = gr.Textbox(label="Camera ID to remove")
                    del_btn = gr.Button("Remove Camera")
                    refresh_btn = gr.Button("Refresh Cameras")
                    cams_dd = gr.Dropdown(choices=self.list_cameras(), label="Select Camera")
                    start_btn = gr.Button("Start Stream")
                    stop_btn = gr.Button("Stop Stream")

                    gr.Markdown("### Register new person")
                    name_in = gr.Textbox(label="Person Name")
                    files = gr.File(file_count="multiple", label="Upload images (multiple)")
                    reg_btn = gr.Button("Register Person")

                    gr.Markdown("### Export logs")
                    export_btn = gr.Button("Export CSV")
                    export_out = gr.File()

                with gr.Column(scale=2):
                    live_out = gr.Image(label="Live feed (select camera & press Start)", shape=(480,640))

            # callbacks
            add_btn.click(fn=lambda cid, src: self.add_camera(cid, src), inputs=[cam_id, cam_src], outputs=[])
            del_btn.click(fn=lambda cid: self.remove_camera(cid), inputs=[del_id], outputs=[])
            refresh_btn.click(fn=lambda: gr.update(choices=self.list_cameras()), outputs=[cams_dd])
            # start stream -> use generator as output; gr.Button.click supports generator functions
            def start_stream_gen(selected_cam):
                if selected_cam is None:
                    # yield blanks
                    while True:
                        yield np.zeros((480,640,3), dtype=np.uint8)
                else:
                    yield from self.stream_generator(selected_cam)

            start_btn.click(fn=start_stream_gen, inputs=[cams_dd], outputs=[live_out])
            stop_btn.click(fn=lambda c: self.stop_stream(c), inputs=[cams_dd], outputs=[])

            reg_btn.click(fn=lambda n, fls: self.register_person(n, fls), inputs=[name_in, files], outputs=[])

            export_btn.click(fn=lambda: self.export_logs(), outputs=[export_out])

        demo.launch(server_name=host, server_port=port, share=share)
