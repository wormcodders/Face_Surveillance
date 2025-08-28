from src.gui import GUI

if __name__ == "__main__":
    # Toggle GPU usage for both detector and recognizer
    USE_GPU = False   # <--- set True if you have CUDA + onnxruntime-gpu installed

    gui = GUI(
        model_face="models/yolov8n-face.pt",        # .pt (ultralytics) recommended for dev
        model_rec="models/arcface_resnet100.onnx", # ArcFace ONNX
        db_dir="database",
        use_gpu=USE_GPU
    )
    gui.launch()
